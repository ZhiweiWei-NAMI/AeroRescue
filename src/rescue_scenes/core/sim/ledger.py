"""Single-entry resource accounting (authored, uncalibrated cost models)."""
from copy import deepcopy
import math

COST_KEYS = ("time_s", "energy_Wh", "bytes", "cpu_s")


def cost_record(cost=None):
    """Return a finite, nonnegative four-resource record."""
    result = {}
    for key in COST_KEYS:
        value = float((cost or {}).get(key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("资源成本必须为有限非负数 / resource cost must be finite and nonnegative")
        result[key] = int(value) if key == "bytes" else value
    return result


class Ledger:
    def __init__(self, sim):
        self.sim = sim
        self.entries = []
        self.by_agent = {}
        self.by_task = {}
        self.totals = cost_record()
        self._seen = set()

    def add(self, instance, agent, action, cost, task_ref=None):
        """Charge each unique work instance once, including failed/canceled work."""
        if instance in self._seen:
            return None
        cost = cost_record(cost)
        self._seen.add(instance)
        task_id = task_ref[0] if task_ref else None
        row = dict(instance=instance, agent=agent, action=action, **cost)
        row.update(task=task_id, task_rev=task_ref[1] if task_ref else None)
        self.entries.append(row)
        buckets = [self.totals, self.by_agent.setdefault(agent, cost_record())]
        if task_id is not None:
            buckets.append(self.by_task.setdefault(task_id, cost_record()))
        for bucket in buckets:
            for key in COST_KEYS:
                bucket[key] += cost[key]
        runtime = self.sim.agents.get(agent)
        if runtime and cost["energy_Wh"] and not runtime.spec.get("mains_power", runtime.spec.get("mobility") == "none"):
            eid = runtime.platform_eid
            battery = self.sim.world.get(eid, "platform.battery_remaining_Wh")
            if battery is not None and math.isfinite(float(battery)):
                self.sim.write_truth(eid, "platform.battery_remaining_Wh",
                                     max(0.0, float(battery) - cost["energy_Wh"]),
                                     "D04", actor=agent, task_ref=task_ref,
                                     action_instance=instance, visible_to=[agent])
        self.sim.emit(layer="action", family="cost.recorded", kind="cost.recorded",
                      grounding="catalog", refs=["F239", "D04"] + ([action] if action.startswith("A") else []),
                      subject=instance, actor=agent, task=task_id,
                      task_rev=task_ref[1] if task_ref else None,
                      action_instance=instance, cost=cost, after=row,
                      visible_to=[agent] if agent else ["*truth*"],
                      text_zh="统一账本记录实际消耗，失败与取消也保留成本。")
        return row

    def to_dict(self):
        return deepcopy(dict(entries=self.entries, totals=self.totals,
                             by_agent=self.by_agent, by_task=self.by_task))
