"""PX4 embodiment bridge: native telemetry -> rescue truth, A001 goals -> PX4 commands.

The rescue engine is the single writer of its entities. This module converts
only; a caller applies ``sync`` on the engine thread before publication, and
submits the returned command rows as ordinary child commands.

Command schema IDs follow the backend name the installed AAS adapter registers
(``adapters.<backend>.arm`` ...), which is not necessarily ``px4``; a caller may
override the prefix explicitly (``prefix=`` / ``px4_command_prefix`` option).
"""
import math

ACTIONS=('arm','takeoff','goto','hold','land')


def px4_backend():
    """The registered AAS px4-gazebo backend name, with a plain fallback."""
    try:
        from aeroagentsim.adapters.schemas import PX4_GAZEBO
        return str(PX4_GAZEBO)
    except (ImportError,AttributeError):
        return 'px4'


PX4_BACKEND=px4_backend()
PX4_PREFIX=f'adapters.{PX4_BACKEND}'
PX4_SCHEMAS=frozenset(f'{PX4_PREFIX}.{name}' for name in ACTIONS)
ARM,TAKEOFF,GOTO,HOLD,LAND=(f'{PX4_PREFIX}.{name}' for name in ACTIONS)
MOVEMENT_ACTIONS=frozenset({'A001','A021'})
# 2D arrival tolerance for a PX4-flown movement leg.
ARRIVAL_TOLERANCE_M=2.0
# Native AAS px4-gazebo field names, keyed by the short semantic names.
FIELDS={'position':'position_enu','velocity':'velocity_enu','attitude':'attitude_quat',
        'battery':'battery','armed':'armed','mode':'flight_mode','landed':'landed_state'}
CAUSE='PX4-EMBODIMENT'
MOVING_SPEED_M_S=0.5
# MAVSDK LandedState names arrive upper-case from the native service.
LANDED_STATES=frozenset({'on_ground','ground','landed','landing'})


def command_schemas(prefix=None):
    """Command schema IDs for one backend prefix (default: the installed one)."""
    base=prefix or PX4_PREFIX
    return frozenset(f'{base}.{name}' for name in ACTIONS)


def enu_yaw_to_compass_deg(yaw_rad):
    """ENU yaw (east=0, counter-clockwise) -> MAVSDK compass yaw (north=0, clockwise)."""
    if not math.isfinite(yaw_rad):
        raise ValueError('PX4 yaw must be finite')
    return (90.0-math.degrees(yaw_rad))%360.0


def compass_deg_to_enu_yaw(yaw_deg):
    """Inverse of :func:`enu_yaw_to_compass_deg`; radians in (-pi, pi]."""
    if not math.isfinite(yaw_deg):
        raise ValueError('PX4 compass yaw must be finite')
    return math.radians(((90.0-float(yaw_deg))+180.0)%360.0-180.0)


def yaw_from_quaternion(quat):
    """ENU heading of an ENU attitude quaternion ``[x, y, z, w]``."""
    x,y,z,w=(float(value) for value in quat)
    return math.atan2(2.0*(w*z+x*y),1.0-2.0*(y*y+z*z))


def native_sample(telemetry):
    """Normalize one telemetry row to AAS px4-gazebo field names (idempotent)."""
    if not isinstance(telemetry,dict):
        raise TypeError('PX4 telemetry must be a mapping')
    return {FIELDS.get(key,key):value for key,value in telemetry.items()}


def motion_state(telemetry):
    """Map armed/mode/landed/velocity to the catalog motion enum."""
    sample=native_sample(telemetry)
    if not sample.get('armed'): return 'idle'
    landed=sample.get('landed_state')
    if landed is True or (isinstance(landed,str) and landed.strip().lower() in LANDED_STATES):
        return 'landed'
    velocity=sample.get('velocity_enu') or (0.0,0.0,0.0)
    return 'moving' if math.hypot(float(velocity[0]),float(velocity[1]))>=MOVING_SPEED_M_S else 'holding'


def _command(schema,payload,target):
    return {'kind':'command','schema':schema,'target':target,'payload':payload}


def movement_commands(entity_id,goal_enu,yaw_rad,*,takeoff_altitude_m=12.0,land=False,
                      airborne=False,target='px4',prefix=None):
    """Pure A001 goal (ENU + ENU yaw) -> arm/takeoff/goto/hold|land sequence.

    ``airborne`` means the vehicle is already flying: neither arm nor takeoff is
    re-issued, so a sequence of waypoints never repeats the takeoff.
    """
    goal=[float(value) for value in goal_enu]
    if len(goal)!=3 or not all(math.isfinite(value) for value in goal):
        raise ValueError('PX4 movement goal requires a finite 3D ENU position')
    altitude=float(takeoff_altitude_m)
    if not math.isfinite(altitude) or altitude<=0:
        raise ValueError('PX4 takeoff altitude must be positive and finite')
    base=prefix or PX4_PREFIX
    rows=[]
    if not airborne:
        rows.append(_command(f'{base}.arm',{'entity':entity_id},target))
        rows.append(_command(f'{base}.takeoff',{'entity':entity_id,'altitude_m':altitude},target))
    rows.append(_command(f'{base}.goto',{'entity':entity_id,'position_enu':goal,
                                        'yaw_deg':enu_yaw_to_compass_deg(float(yaw_rad))},target))
    rows.append(_command(f'{base}.{"land" if land else "hold"}',{'entity':entity_id},target))
    return rows


def complete_reached_movements(sim,entities,tolerance_m=ARRIVAL_TOLERANCE_M,*,positions_enu=None):
    """Settle active external legs using core XY and explicit ENU altitude.

    A 2D catalog goal uses horizontal distance. A goal with altitude requires a
    full native pose and a 3D distance check; a ground projection cannot prove
    vertical arrival. Completion always uses the normal kernel result path.
    """
    tolerance=float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('arrival tolerance must be finite and nonnegative')
    done=[]
    active=getattr(sim,'_active_instances',sim.instances)
    for entity in entities:
        agent=sim.agents.get(entity)
        platform=getattr(agent,'platform_eid',entity) if agent is not None else entity
        pose=sim.world.get(platform,'entity.pose')
        if not isinstance(pose,(list,tuple)) or len(pose)<2:
            continue
        native=(positions_enu or {}).get(platform)
        if isinstance(native,dict):
            native=native.get('position_enu')
        for iid,rec in sorted(list(active.items())):
            if (rec.get('status')!='running' or not rec.get('external_motion')
                    or rec.get('cancel_requested') or rec.get('action_id') not in MOVEMENT_ACTIONS):
                continue
            owner=sim.agents.get(rec.get('agent_id'))
            if rec.get('agent_id')!=entity and getattr(owner,'platform_eid',None)!=platform:
                continue
            params=rec.get('params') or {}
            goal=(params.get('_motion') or {}).get('goal')
            if not isinstance(goal,(list,tuple)) or len(goal)<2:
                continue
            explicit=params.get('goal_enu') or params.get('goal')
            target=explicit if isinstance(explicit,(list,tuple)) and len(explicit)==3 else goal
            values=[float(pose[0]),float(pose[1]),float(goal[0]),float(goal[1])]
            distance=math.hypot(values[0]-values[2],values[1]-values[3])
            if len(target)==3:
                position=native if isinstance(native,(list,tuple)) and len(native)==3 else pose
                if len(position)!=3:
                    continue
                altitude,goal_altitude=float(position[2]),float(target[2])
                values.extend((altitude,goal_altitude))
                distance=math.hypot(distance,altitude-goal_altitude)
            if all(map(math.isfinite,values)) and distance<=tolerance:
                if sim.complete_external_action(iid):
                    done.append(iid)
    return done


class PX4EmbodimentBridge:
    """Bidirectional adapter between a rescue world and the PX4 container.

    ``vehicles`` maps rescue entity ids to PX4 container vehicle names; an
    unmapped vehicle name is treated as the entity id itself. ``pose`` retains
    the last full 3D ENU position and attitude so publication can use the real
    altitude and heading instead of the scripted display offset.
    """

    def __init__(self,world,vehicles=None):
        self.world=world
        self.sim=world.sim
        # RescueWorld owns the canonical grid; a bare Environment starts at the
        # scene origin, so ENU coordinates are already core coordinates there.
        self.grid=getattr(world,'grid',None)
        self.vehicles=dict(vehicles or {})
        if len(set(self.vehicles.values()))!=len(self.vehicles):
            raise ValueError('px4_vehicles must map each PX4 vehicle to one entity')
        self.entity_by_vehicle={name:entity for entity,name in self.vehicles.items()}
        self.pose={}
        self.last={}

    def entity(self,vehicle):
        return self.entity_by_vehicle.get(vehicle,vehicle)

    def _write(self,entity,key,value):
        self.sim.world.set(entity,key,value,{'rule':CAUSE,'t':self.sim.t,'scripted':False})

    def apply_telemetry(self,entity,telemetry):
        """Write one native sample into the entity's truth fields; return applied keys."""
        if entity not in self.sim.world.entities:
            raise ValueError(f'PX4 telemetry names an unknown entity: {entity}')
        sample=native_sample(telemetry)
        applied={}
        pose=self.pose.setdefault(entity,{'position_enu':None,'attitude':None,'yaw_rad':None})
        if 'position_enu' in sample:
            position=[float(value) for value in sample['position_enu']]
            if len(position)!=3 or not all(math.isfinite(value) for value in position):
                raise ValueError('PX4 position_enu requires a finite 3D ENU position')
            xy=self.grid.core(position[:2]) if self.grid is not None else position[:2]
            core=[float(xy[0]),float(xy[1])]
            self._write(entity,'entity.pose',core)
            pose['position_enu']=position
            applied['entity.pose']=core
            applied['position_enu']=position
        if 'attitude_quat' in sample:
            attitude=[float(value) for value in sample['attitude_quat']]
            if len(attitude)!=4 or not all(math.isfinite(value) for value in attitude):
                raise ValueError('PX4 attitude_quat requires four finite components')
            pose['attitude']=attitude
            pose['yaw_rad']=yaw_from_quaternion(attitude)
            applied['attitude_quat']=attitude
        elif sample.get('yaw_deg') is not None:
            pose['yaw_rad']=compass_deg_to_enu_yaw(float(sample['yaw_deg']))
        if 'battery' in sample:
            battery=sample['battery']
            fraction=float(battery['remaining_fraction'] if isinstance(battery,dict) else battery)
            if not 0.0<=fraction<=1.0:
                raise ValueError('PX4 battery remaining_fraction must be in [0,1]')
            capacity=self.sim.world.get(entity,'platform.battery_capacity_Wh')
            if capacity is None:
                raise ValueError(f'PX4 vehicle entity has no battery capacity: {entity}')
            wh=round(fraction*float(capacity),6)
            self._write(entity,'platform.battery_remaining_Wh',wh)
            applied['platform.battery_remaining_Wh']=wh
        if {'armed','landed_state','velocity_enu'} & set(sample):
            state=motion_state({**self.last.get(entity,{}),**sample})
            self._write(entity,'platform.motion_state',state)
            applied['platform.motion_state']=state
        self.last[entity]={**self.last.get(entity,{}),**sample}
        return applied

    def sync(self,samples):
        """Apply every sample; the caller publishes truth afterwards."""
        applied=[]
        for sample in samples:
            entity=self.entity(sample.get('vehicle'))
            applied.append(self.apply_telemetry(entity,sample))
        return applied

    def commands_for_action(self,entity,action_id,params,*,target='px4',airborne=False,
                            land=False,prefix=None):
        """A001/A021 params (``goal`` ENU) -> PX4 command rows.

        Without an explicit ``yaw_rad``/``yaw_deg`` the heading follows the
        rescue engine's own convention: the ENU bearing from the current pose.
        """
        if action_id not in MOVEMENT_ACTIONS:
            raise ValueError(f'PX4 bridge forwards movement actions only, not {action_id}')
        goal=params.get('goal') or params.get('goal_enu')
        if goal is None:
            raise ValueError('movement action requires a goal ENU position')
        yaw=params.get('yaw_rad')
        if yaw is None and params.get('yaw_deg') is not None:
            yaw=compass_deg_to_enu_yaw(params['yaw_deg'])
        if yaw is None:
            pose=self.sim.world.get(entity,'entity.pose')
            current=self.grid.enu(pose[:2]) if self.grid is not None else [*pose[:2],0.0]
            yaw=math.atan2(float(goal[1])-current[1],float(goal[0])-current[0])
        return movement_commands(entity,goal,yaw,target=target,airborne=airborne,land=land,
                                 takeoff_altitude_m=params.get('takeoff_altitude_m',12.0),
                                 prefix=prefix)
