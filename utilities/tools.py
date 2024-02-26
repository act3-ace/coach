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

from typing import Any


class MultiDict:
    def __init__(self, tags: dict = dict()):
        self.tags = tags # tags - str(type): list(values)
        self._vals = dict()
        self._refs = {tag: {val:[] for val in values} for tag, values in tags.items()}

    def __getattr__(self, __name: str) -> Any:
        if __name in self.tags.keys():
            return SliceMakerDict(__name, self)
        
    def __getitem__(self, select_name: dict):
        new_tags = dict(self.tags)

        for tag_name in select_name.keys():
            del new_tags[tag_name]

        new_mdct = MultiDict(new_tags)

        items = [set(self._refs[tag_name][tag]) for tag_name, tag in select_name.items()]
        items = set.intersection(*items)
        
        for item in items:
            new_tag = dict(self._vals[item])
            for tag_type_name in select_name.keys():
                del new_tag[tag_type_name]

            new_mdct[new_tag] = item

        return new_mdct


    def __setitem__(self, key: dict, value: Any):
        # Following are sufficient to check all keys supplied:
        if not len(key.keys()) == len(self.tags.keys()):
            raise Exception(f"Mismatched key lengths between {key.keys()} and {self.tags.keys()}")

        for tag_type, tag in key.items():
            try:
                tags = self._refs[tag_type]
            except:
                raise Exception(f"Trying to set tag_type {tag_type} but this is not found in {self.tags.keys()}")
            
            if tag not in tags:
                self.tags[tag_type].append(tag)
                tags[tag] = list()

            tags[tag].append(value)

        self._vals[value] = key

    def __repr__(self) -> str:
        s = ""
        tab_count = []
        for tag_type in self.tags.keys():
            count = int(np.floor(len(tag_type) / 8)) + 1
            tab_count.append(count)
            s += f"{tag_type}\t"

        s += "value: \n"

        for item, tag in self._vals.items():

            # Keep the same order as above
            for i, tag_type in enumerate(self.tags.keys()):
                max_len = tab_count[i]*8 - 2
                tag_len = len(str(tag[tag_type]))

                if tag_len > max_len:
                    sp = str(tag[tag_type])[tab_count[i]*8-2] + "\t"
                else:
                    count = int(np.floor((max_len - tag_len)/8) + 1)
                    sp = str(tag[tag_type]) + "\t"*count

                s += sp

            s += str(item)
            s += "\n"

        return s
        
class SliceMakerDict:
    def __init__(self, tag_type_name, parent: MultiDict):
        self.parent: MultiDict = parent
        self.tag_type_name = tag_type_name

    def __repr__(self) -> str:
        return str(self.parent.tags[self.tag_type_name])
    
    def __getitem__(self, key):
        new_tags = dict(self.parent.tags )
        del new_tags[self.tag_type_name]

        new_mdct = MultiDict(new_tags)
        for item, tag in self.parent._vals.items():
            if tag[self.tag_type_name] == key:
                new_tag = dict(tag)
                del new_tag[self.tag_type_name]

                new_mdct[new_tag] = item

        return new_mdct





def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def plot_traj(p, q, v, traj=None, w=[0, 0], ax=None):
    if ax is None:
        ax = plt.gca()
    if not traj is None:
        ax.plot(traj[:, 0], traj[:, 1], "--")
    ax.plot(p[0], p[1], "or")
    ax.plot(q[0], q[1], "og")
    ax.quiver(p[0], p[1], v[0], v[1], angles="xy", scale_units="xy", scale=1)
    ax.quiver(
        p[0], p[1], float(w[0]), float(w[1]), angles="xy", scale_units="xy", scale=1
    )
    # ax.set_xlim(-60,10)
    # ax.set_ylim(-10,10)
    return ax


def plot_traj_3d(p, q, v, traj=None, w=[0, 0], ax=None):
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    if not traj is None:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "--")
    ax.plot(p[0], p[1], p[2], "or")
    ax.plot(q[0], q[1], q[2], "og")
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2])
    ax.quiver(p[0], p[1], p[2], float(w[0]), float(w[1]), float(w[2]))
    return ax


def r(t, w, p, v):
    return p + (t**2) * w / 2 + t * v


def r_p(t, w, v):
    return t * w + v


def g(t, w, t1, p, v):
    if t < t1:
        return r(t, w, p, v)
    else:
        return r(t1, w, p, v) + (t - t1) * r_p(t1, w, v)


def dg(t, w, t1, p):
    return r(t1, w, p) + t * r_p(t1, w, p)


def rotation_matrix(theta):
    theta

    rotMatrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    return rotMatrix
