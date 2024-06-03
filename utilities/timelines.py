# Copyright (c) 2024 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import json
import copy

from typing import Any


## Aux Classes:
# Note: Everything here has been written so that it (at present)
# deepcopies. If you do anything to make a model not deepcopy
# by default you'll need to add a __deepcopy__ function

# %%
#######################
class TimelineEvent:
    id_idx = 0
    @staticmethod
    def from_dict(event):
        tmp = TimelineEvent(
            event["label"], 
            event["parameters"],
            tags = {k:v for k,v in event.items() if (k not in {"label", "parameters", "id"})},
            id = event.get("id", None)
            )
        
        return tmp

    def __init__(self, label, parameters, tags: dict = dict(), id=None):
        if id == None:
            self.id = TimelineEvent.id_idx
            TimelineEvent.id_idx += 1

        self.label = label
        self.parameters = parameters
        self.tags = tags

    def to_dict(self):
        return {
            "label": self.label,
            "parameters": self.parameters,
            "tags": self.tags
        }

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"label: {self.label}, params: {self.parameters}, tags: {self.tags}"
    
    def copy(self):
        return copy.copy(self)


class TimelineInterval(TimelineEvent):
    def __init__(self, *args, extent=1):
        super().__init__(*args)
        self.extent = extent
        self.range = range(self.start, self.start + extent + 1)


class Timeline:
    def __init__(self, length=0, fixedlength=None):
        self.timeline = dict()
        self.events = dict()
        self.labels = dict()

        self.fixedlength = False
        self.length = length
        self.allow_dup_labels = False

        if fixedlength is not None:
            if length>0:
                raise Exception("Both length and fixedlength are specified. Only one should be.")
            self.fixedlength = True
            self.length = fixedlength

    def __repr__(self) -> str:
        return f"Timeline: {str(self.events)}"
    
    def __len__(self):
        return self.length

    def __deepcopy__(self, memo):
        tmp = Timeline()
        tmp.__dict__ = copy.deepcopy(self.__dict__)
        return(tmp)
    
    def add_events_from_dict(self, events):
        # Expected format: 
        # event = {
        #     "start": int,
        #     "action": str,
        #     "parameters": np.array,
        #     "role": str,
        # }

        for event in events:
            self.add_event(event["start"], TimelineEvent.from_dict(event))

    
    def add_events(self, events):
        if type(events) is TimelineEvent:
            self.add_event(events)
        else:
            for t, e in events:
                self.add_event(t, e)
        
    def add_event(self, time, event: TimelineEvent):
        if self.fixedlength:
            if time > self.length:
                raise Exception(f"Timeline has fixed length {self.length} but event added at {time}.")
        else:
            self.length = max(self.length, time)

        if not self.allow_dup_labels:
            mark_for_delete = []
            for e in self.timeline.get(time, []):
                if e.label == event.label:
                    mark_for_delete.append(e)
            
            for e_del in mark_for_delete:
                self.delete_event_at(time=time, event=e_del)

                
        if time not in self.timeline.keys():
            self.timeline[time] = set()

        if event not in self.events.keys():
            self.events[event] = set()

        if event.label not in self.labels.keys():
            self.labels[event.label] = set()

        self.timeline[time].add(event)
        self.timeline = dict(sorted(self.timeline.items()))

        self.events[event].add(time)
        self.labels[event.label].add((time, event))

    def delete_event_at(self, event, time):
        self.events[event].remove(time)
        if len(self.events[event]) == 0:
            del self.events[event]

        self.timeline[time].remove(event)
        if len(self.timeline[time]) == 0:
            del self.timeline[time]
            if time == self.length-1 and not self.fixedlength:
                self.length = max(self.timeline.keys())
        
        self.labels[event.label].remove((time, event))
        if len(self.labels[event.label]) == 0:
            del self.labels[event.label]

    def delete_all_occurences(self, event):
        while event in self.event.keys():
            t = list(self.events[event])[0]
            self.delete_event_at(event=event, time=t)

    def move_events(self, time, to, events=None):
        if events is None:
            events = copy.copy(self.timeline[time]) # we're modifying the object as we iterate through it
        
        for event in events:
            self.delete_event_at(event = event, time=time)
            self.add_event(event=event, time=to)
            

    
    def add_interval(self, event: TimelineInterval):
        for t in event.range:
            self.add_event(t, event)

    def first(self, event):
        return(min(self.events[event]))
    
    def last(self, event):
        return(max(self.events[event]))
    
    def interval_start(self, event, t):
        if t not in self.events[event]:
            return None
        
        while t in self.events[event]:
            t += -1

        return t + 1
    
    def interval_end(self, event, t):
        if t not in self.events[event]:
            return None
        
        while t in self.events[event]:
            t += 1

        return t - 1

    def get_all_labeled(self, label):
        return self.labels[label]
    
    def get_events_at_time(self, time):
        return self.timeline[time]
    
    def get(self, time, label):
        r = []

        if label not in self.labels.keys():
            return None
            # raise Exception(f"'{label}' not found in known labels: {list(labels.keys())}")

        for t, event in self.labels[label]:
            if t == time:
                r.append(event)
        
        if len(r) == 1:
            return r[0]
        
        if len(r) == 0:
            return None
        
        if len(r) > 1:
            raise Exception(f"Multiple events with same label at same time: {r}")
        
    def itterate_over_label(self, label):
        tmp = dict()
        for time, event in self.labels[label]:
            if time not in tmp.keys():
                tmp[time] = []
            
            tmp[time].append(event)
        
        # Returns a key sorted list of tuples (t, [e1, e2,...])
        return iter(sorted(tmp.items()))
    
    def next_event(self, time):
        L = list(self.timeline.keys())
        if len(L) == 0:
            return None, None
        
        last_high = L[-1]
        if time >= last_high:
            return None, None
        if time < L[0]:
            return L[0], self.timeline[L[0]]
        
        # Go through timeline vis bisection. Probably overkill
        while len(L)>1:
            half_idx = int(len(L)/2)
            
            if L[half_idx] > time:
                last_high = L[half_idx]
                L = L[:half_idx]
            else:
                L = L[half_idx:]
                if L[-1] > time:
                    last_high = L[-1]

        return last_high, self.timeline[last_high]
        
    def __iter__(self):
        ## Should already be sorted
        return iter(sorted(self.timeline.items()))
    
    def add_timeline_to(self, timeline, allow_dup_labels=True):
        ## Add new coa elements
        for event, times in timeline.events.items():
            if event not in self.events.keys():
                self.events[event] = set()

            to_update = times - self.events[event]

            for t in to_update:
                self.add_event(time=t, event=event)

    def display(self):
        PADDING = 3
        LINEWIDTH = 64
        LABELLENGTH = 8
        TLSPACE = LINEWIDTH - PADDING - LABELLENGTH

        if self.length < TLSPACE:
            step = 1
            LINEWIDTH = LABELLENGTH + self.length
            TLLENGTH = self.length
        else:
            step = int(np.ceil(self.length/(TLSPACE)))
            TLLENGTH = int(np.ceil(self.length/step))
        
        ## Time:
        ticklabs = [" "]*LINEWIDTH
        ticklabs[0:5] = list("Time:")
        ticks = [" "]*LABELLENGTH + ["-"]*(TLLENGTH+1)

        demarks = list(range(0,self.length, step*10))

        for j, i in enumerate(demarks):
            j = LABELLENGTH + j*10
            
            c = len(str(i))
            tmp = [" "]*7
            if c < 8:
                tmp[(3-int((c-1)/2)):(3-int((c-1)/2)) + c] = str(i)
            else:
                tmp = "{:.2E}".format(c)

            ticklabs[j-3:j+4] = tmp

            ticks[j] ="|"
        
        ticks[-1] ="|"
            
        ## Lines

        lines = {
            label: list(str(label)[:LABELLENGTH]) + [" "] * max(0,LABELLENGTH - len(label)) + ["·"]*TLLENGTH
            for label in self.labels.keys()
            }

        for label, events in self.labels.items():
            for t, event in events:
                lines[label][LABELLENGTH + int(t/step)] = "o"
        
        lines = ["".join(line) for line in lines.values()]
        lines.insert(0, "".join(ticklabs))
        lines.insert(1, "".join(ticks))

        return "\n".join(lines)

    
class Timelines:
    def __init__(self, labels, length=0, fixedlength = None):
        self.timelines = {label:Timeline(length=length, fixedlength=fixedlength) for label in labels}
        
    def get_events(self, time):
        return {label: tl.get_events_at_time(time) for label, tl in self.timeslines.items()}

    def __repr__(self) -> str:
        return str(self.timelines)
    
    def __getattr__(self, __name):
        return self.timelines[__name]

    # def __deepcopy__(self, memo):
    #     tmp = self.__class__(labels=[])
    #     tmp.timelines = copy.deepcopy(self.timelines)
    #     print("Copying Timeline", tmp.timelines)

    #     return(tmp)

    def __getstate__(self):
        return self.timelines

    def __setstate__(self, d):
        self.timelines = d
