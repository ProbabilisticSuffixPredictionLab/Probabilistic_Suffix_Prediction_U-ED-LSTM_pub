#!/usr/bin/env python3

from enum import Enum
import numpy as np
import random
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from datetime import datetime, timedelta


class Resource:
    def __init__(self, name):
        self.name = name

    def sample_duration(self, activity, simulator):
        raise NotImplementedError
    
    def __str__(self):
        return str(self.name)


class DefaultResource(Resource):
    """
    Simple normal distributed task durations
    """
    def sample_duration(self, activity, simulator):
        return 1

class ActivityTypes(Enum):
    RECEPTION = 0
    RECEIVING_INSPECTION = 1
    DISASSEMBLY = 2
    REPAIR = 3
    QUALITY_CONTROL = 4
    CREATE_INVOICE = 5
    SHIPPING = 6
    FINISHED = 7


class ActivityState(Enum):
    ACTIVATED = 'SCHEDULE'
    STARTED = 'START'
    COMPLETED = 'COMPLETE'

    def __str__(self):
        return str(self.value)


class Activity:
    def __init__(self, id, type, instance):
        self.id = id
        self.type = type
        self.state = ActivityState.ACTIVATED
        self.instance = instance
        self.resource = None

    def start(self, resource):
        self.state = ActivityState.STARTED
        self.resource = resource

    def complete(self):
        self.state = ActivityState.COMPLETED

    def __str__(self):
        return self.type.name

class ControlFlow:
    def __init__(self, instance):
        self.instance = instance
        self.iterations = random.randint(1, 7)
        self.seldom_forgotten_receiving_inspection = lambda : random.choices([True, False], weights=[0.1, 0.9])[0]
        self.often_forgotten_receiving_inspection = lambda : random.choices([True, False], weights=[0.4, 0.6])[0]
        self.seldom_forgotten_quality_control = lambda : False if self.instance.simulator.current_time < 365 else \
                                         random.choices([True, False], weights=[0.1, 0.9])[0]
        self.often_forgotten_quality_control = lambda : False if self.instance.simulator.current_time < 365 else \
                                         random.choices([True, False], weights=[0.4, 0.9])[0]
        self.forgotten_quality_control = None

    def first_activity_type(self):
        return ActivityTypes.RECEPTION

    def next_activity_type(self, current_activity_type, last_resource):
        match current_activity_type:
            case ActivityTypes.RECEPTION:
                self.forgotten_receiving_inspection = (self.often_forgotten_receiving_inspection()
                                                       if last_resource.name in [1,4] 
                                                       else self.seldom_forgotten_receiving_inspection())
                if self.forgotten_receiving_inspection:
                    return ActivityTypes.DISASSEMBLY
                else:
                    return ActivityTypes.RECEIVING_INSPECTION
            case ActivityTypes.RECEIVING_INSPECTION:
                if self.forgotten_receiving_inspection:
                    return ActivityTypes.CREATE_INVOICE
                else:
                    return ActivityTypes.DISASSEMBLY
            case ActivityTypes.DISASSEMBLY:
                return ActivityTypes.REPAIR
            case ActivityTypes.REPAIR:
                self.iterations -= 1
                if self.forgotten_quality_control == None:
                    self.forgotten_quality_control = (self.often_forgotten_quality_control()
                                                           if last_resource.name in [2,5]
                                                           else self.seldom_forgotten_quality_control())
                if self.forgotten_receiving_inspection:
                    return ActivityTypes.QUALITY_CONTROL
                else:
                    if self.forgotten_quality_control:
                        if self.iterations:
                            return ActivityTypes.REPAIR
                        else:
                            return ActivityTypes.CREATE_INVOICE
                    else:
                        return ActivityTypes.QUALITY_CONTROL
            case ActivityTypes.QUALITY_CONTROL:
                if self.forgotten_receiving_inspection:
                    if self.iterations == 0:
                        return ActivityTypes.RECEIVING_INSPECTION
                    else:
                        return ActivityTypes.REPAIR
                else:
                    if self.iterations == 0:
                        return ActivityTypes.CREATE_INVOICE
                    else:
                        return ActivityTypes.REPAIR         
            case ActivityTypes.CREATE_INVOICE:
                return ActivityTypes.SHIPPING
            case ActivityTypes.SHIPPING:
                return ActivityTypes.FINISHED
        print('obacht')


class ProcessInstance:

    def __init__(self, id, simulator):
        self.id = id
        self.simulator = simulator
        self.control_flow = ControlFlow(self)
        self.current_activity_id = 0
        self.current_activity = None
        self.current_resource = None
        self.activities = []
        self.resources = []
        self.finished = False

    def start_instance(self):
        first_activity_type = self.control_flow.first_activity_type()
        self.current_activity = Activity(0, first_activity_type, self)
        self.activities.append(self.current_activity)

    def activate_next_activity(self, last_resource):
        self.current_resource = last_resource
        if not self.finished:
            next_activity_type = self.control_flow.next_activity_type(self.current_activity.type, self.current_resource)
            self.current_activity_id += 1
            self.current_activity = Activity(self.current_activity_id, next_activity_type, self)
            self.activities.append(self.current_activity)
        if self.current_activity.type == ActivityTypes.FINISHED:
            self.finished = True

    def has_finished(self):
        return self.finished
    
    def __str__(self):
        return str(self.id)


class EventType(Enum):
    INSTANCE_SPAWN = 0
    ACTIVITY_ACTIVATE = 1
    ACTIVITY_START = 2
    ACTIVITY_COMPLETE = 3
    INSTANCE_COMPLETE = 4


class Event:
    def __init__(self, event_type, event_time, data):
        self.type = event_type
        self.time = event_time
        self.data = data


class Resources:
    def __init__(self, simulator):
        self.resources = [DefaultResource(1),
                          DefaultResource(2),
                          DefaultResource(3),
                          DefaultResource(4),
                          DefaultResource(5)
        ]
        self.idle_resources = self.resources.copy()
        self.working_resources = []
        self.simulator = simulator

    def eligible(self, activity, resource):
        # TODO
        return True

    def allocate(self, resource):
        self.working_resources.append(resource)
        self.idle_resources.remove(resource)

    def free(self, resource):
        self.idle_resources.append(resource)
        self.working_resources.remove(resource)

    def sample_duration(self, activity, resource):
        return resource.sample_duration(activity, self.simulator)


class ProcessSimulator:
    def __init__(self, start_time = 0, logger = None):
        self.current_time = start_time
        self.max_process_instance_id = -1
        self.event_queue = []
        self.activated_activities = []
        self.logger = logger
        self.resources = Resources(self)
        
        self.spawn_instance()
    

    def spawn_instance(self):
        self.max_process_instance_id += 1
        next_instance = ProcessInstance(self.max_process_instance_id, self)
        next_instance_spawn_time = self.current_time + np.random.exponential(scale = 24)
        next_instance_spawn_event = Event(EventType.INSTANCE_SPAWN,
                                          next_instance_spawn_time,
                                          {'instance' : next_instance})
        self.event_queue.append(next_instance_spawn_event)

    def activate_activity(self, activity):
        activate_event = Event(EventType.ACTIVITY_ACTIVATE,
                                self.current_time,
                                {'activity' : activity})
        self.event_queue.append(activate_event)

    def start_activity(self, activity, resource):
        start_activity = Event(EventType.ACTIVITY_START,
                               self.current_time,
                               {'activity' : activity, 'resource' : resource})
        self.event_queue.append(start_activity)

    def complete_activity(self, activity, resource, activity_completed_time):
        activity_completed = Event(EventType.ACTIVITY_COMPLETE,
                                activity_completed_time,
                                {'activity' : activity, 'resource' : resource})
        self.event_queue.append(activity_completed)

    def _instance_spawned(self, event):
        # activate first activity
        event.data['instance'].start_instance()
        self.activate_activity(event.data['instance'].current_activity)
        # set next instance spawn
        self.spawn_instance()

    def _activity_activated(self, event):
        self.activated_activities.append(event.data['activity'])

    def _activity_started(self, event):
        event.data['activity'].start(event.data['resource'])

    def _activity_completed(self, event):
        event.data['activity'].complete()
        self.resources.free(event.data['resource'])
        event.data['activity'].instance.activate_next_activity(event.data['resource'])
        if not event.data['activity'].instance.has_finished():
            self.activate_activity(event.data['activity'].instance.current_activity)

    def _instance_completed(self, event):
        pass

    def _start_activated_activities(self):
        restart = False
        for activity in self.activated_activities:
            random.shuffle(self.resources.idle_resources)
            for resource in self.resources.idle_resources:
                if self.resources.eligible(activity, resource):
                    self.resources.allocate(resource)
                    self.activated_activities.remove(activity)
                    # set activity start
                    self.start_activity(activity, resource)

                    # set activity finish
                    duration = self.resources.sample_duration(activity, resource)
                    self.complete_activity(activity, resource, self.current_time + duration)
                    break


    def log_event(self, event):
        if self.logger:
            self.logger.log(event)

    def finish_log(self):
        if self.logger:
            self.logger.finish()

    def simulate(self, max_time = np.inf):
        self.start_time = self.current_time
        while len(self.event_queue):
            if max_time < self.current_time - self.start_time:
                break
            current_event = self.event_queue.pop(0)
            self.current_time = current_event.time

            if current_event.type == EventType.INSTANCE_SPAWN:
                self._instance_spawned(current_event)
            elif current_event.type == EventType.ACTIVITY_ACTIVATE:
                self._activity_activated(current_event)
            elif current_event.type == EventType.ACTIVITY_START:
                self._activity_started(current_event)
            elif current_event.type == EventType.ACTIVITY_COMPLETE:
                self._activity_completed(current_event)
            elif current_event.type == EventType.INSTANCE_COMPLETE:
                self._instance_completed(current_event)
            
            self._start_activated_activities()
            self.event_queue.sort(key = lambda event : event.time)
            self.log_event(current_event)
        self.finish_log()
        return


class PrintLogger:
    def log(self, event):
        activity = event.data['activity'] if event.type not in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else None
        instance = event.data['instance'] if event.type in [EventType.INSTANCE_COMPLETE, EventType.INSTANCE_SPAWN] else event.data['activity'].instance
        resource = event.data['resource'] if event.type in [EventType.ACTIVITY_START] else None
        print(event.time, instance, event.type, activity, resource)

    def finish(self):
        pass

class PandasLogger:
    def __init__(self, out_file):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type == EventType.ACTIVITY_START:
            activity = event.data['activity']
            resource = event.data['resource']
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append([str(activity.instance), str(activity), timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource'])


class XESLogger(PandasLogger):
    def finish(self):
        super().finish()
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)


class XESLifeCycleLogger:
    def __init__(self, out_file='event_log.xes'):
        self.data = []
        self.start_time = datetime(2020, 1, 1)
        self.out_file = out_file

    def log(self, event):
        if event.type in [EventType.ACTIVITY_ACTIVATE, EventType.ACTIVITY_START, EventType.ACTIVITY_COMPLETE]:
            activity = event.data['activity']
            instance = event.data['activity'].instance
            resource = event.data['resource'] if event.type != EventType.ACTIVITY_ACTIVATE else None
            timestamp = self.start_time + timedelta(hours = event.time)
            self.data.append([str(instance), str(activity), activity.state, timestamp, str(resource)])

    def finish(self):
        self.df = pd.DataFrame(self.data, columns = ['case:concept:name', 'concept:name', 'lifecycle:transition', 'time:timestamp', 'org:resource'])
        # Convert DataFrame to Event Log
        log = log_converter.apply(self.df, variant=log_converter.Variants.TO_EVENT_LOG)
        # Export to XES
        pm4py.write_xes(log, self.out_file)

class CSVLogger(PandasLogger):
    def __init__(self, out_file='event_log.csv'):
        super().__init__(out_file)

    def finish(self):
        super().finish()
        self.df.to_csv(self.out_file, index=False)

if __name__ == '__main__':
    #simulator = ProcessSimulator(logger=CSVLogger('test_log.csv'))
    simulator = ProcessSimulator(logger=CSVLogger('repair_shop_train_log.csv'))
    simulator.simulate(24*365)

    simulator = ProcessSimulator(logger=CSVLogger('repair_shop_eval_log.csv'), start_time=365)
    simulator.simulate(24*365)
