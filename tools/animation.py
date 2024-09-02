import pyglet
from abc import ABC, abstractmethod
from typing import Callable, Union
from tools.easing import *

FPS = 60


def lerp1d(start, dest, t):
    return start + (dest - start) * t


class Animation(ABC):
    """
        Blueprint that all animations should follow
        in order to be manipulated by AnimationManager.
    """

    def __init__(self, obj, func_handle: Callable = linear, duration=1):
        self.object = obj
        self.function = func_handle
        self.duration = duration
        self.running = True

    @abstractmethod
    def update(self):
        """ Implement update method for animation."""

    def reset(self):
        """ Implement resetting parameters to initial values."""


class Rotate(Animation):
    """ Rotates object clockwise around Z axis by passed angle in degrees."""

    def __init__(self, obj, angle, func_handle: Callable = linear, duration=1):
        super().__init__(obj, func_handle, duration)
        # adjust anchors to centre for Rectangles
        if isinstance(obj, pyglet.shapes.Rectangle):
            obj.anchor_x, obj.anchor_y = obj.width / 2, obj.height / 2
        self.starting_param_value = None
        self.current_param_value = None
        self.final_param_value = angle
        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time

    def update(self):
        if not self.starting_param_value:
            self.starting_param_value = self.object.rotation
        if not self.current_param_value:
            self.current_param_value = self.object.rotation
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.object.rotation = lerp1d(self.object.rotation, self.final_param_value, t)
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self.object.rotation -= self.final_param_value
        self.current_param_value = None

    def complete(self):
        self.object.rotation = self.final_param_value


class Color(Animation):
    def __init__(self, obj, color_rgb: tuple[int, int, int], func_handle: Callable = linear, duration=1):
        super().__init__(obj, func_handle, duration)
        self.start_param_value = None
        self.current_param_value = None
        self.final_param_value = color_rgb

        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time

    def update(self):
        if not self.start_param_value:
            self.start_param_value = self.object.color
        if not self.current_param_value:
            self.current_param_value = self.object.color
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.current_param_value = (int(lerp1d(self.current_param_value[0], self.final_param_value[0], t)),
                                        int(lerp1d(self.current_param_value[1], self.final_param_value[1], t)),
                                        int(lerp1d(self.current_param_value[2], self.final_param_value[2], t))
                                        )
            self.object.color = self.current_param_value
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self.object.color = self.start_param_value
        self.current_param_value = None

    def complete(self):
        self.object.color = self.final_param_value


class Scale(Animation):
    pass


class Move(Animation):
    def __init__(self, obj, dest_x, dest_y, func_handle: Callable = linear, duration=1):
        """
         Moves pyglet.Sprite or pyglet.Shape object to destination in given time, with Vec2's lerp.
        Lerp parameter "t" here is given as a ratio of elapsed frames to max frames and takes values 0.0 - 1.0.
        Optional easing curve functions are available in easing.py taken from python Arcade.

        :param obj: pyglet.Shape or pyglet.Sprite object to apply animation to
        :param dest_x: x-coordinate of the end point passed object will be moved to
        :param dest_y: y-coordinate of the end point passed object will be moved to
        :param func_handle: Optional easing curve function handle for lerp parameter "t"
        :param duration: desired length of an animation from start to finish in seconds
        """
        super().__init__(obj, func_handle, duration)
        # starting position vector is a constant, first set when animation is executed
        self.start_param_value = None
        # current position vector is a variable, also set first when animation is executed
        self.current_param_value = None
        # set destination position vector from passed arguments
        self.final_param_value = pyglet.math.Vec2(dest_x, dest_y)

        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time

    def update(self):
        if not self.start_param_value:
            self.start_param_value = pyglet.math.Vec2(self.object.x, self.object.y)
        if not self.current_param_value:
            self.current_param_value = pyglet.math.Vec2(self.object.x, self.object.y)
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.object.position = self.current_param_value.lerp(self.final_param_value, t)
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self.object.position = self.start_param_value
        self.current_param_value = None

    def complete(self):
        """ Completes animation by stopping interpolation and assigning final param value."""
        self.object.position = self.final_param_value


class AnimationManagerV3:
    """
        List that stores and logic that manages animations of objects.

        Animations can be added with add() method and will play in sequence.
        If instead a list of animations is passed to add() method, all will
        play simultaneously.

        Multiple managers can be instantiated to play animation sets independently.
        Instant mode manager interrupts previous animation and plays new one immediately.

        Completed:
        Currently all tasks are executed continuously.
        In addition to continuous mode, I should have "sequential" mode
        to jump between adjacent animations with next and previous.
        Color animation added

        Bugs:
        recheck Animaiton/Move class and initial position restoring
        reset ne readi kako treba sa previous() u sequential modu

        Future:
        Ultimately, some kind of "manual" mode would allow animation to  flow
        forward (and possibly backwards!) smoothly when key is pressed.


    """

    def __init__(self, on_completion_func=None, play_mode=0,
                 instant=False, loop=False):
        """ Modes:
                - automatic: continuous
                - manual: sequential
                - manual: seek
        """
        self.on_completion_func = on_completion_func
        self.manager_active = False
        self.task_list = []
        # dict that stores initial position of animation objects for full reset
        self.objects_and_start_params = {}
        self.instant = instant
        self._loop = loop

        self._mode = None
        self.play_mode = play_mode

        self._task_index = 0

    def add(self, task: Union[Animation, list[Animation, ...]]):
        """ Append new task (Animation or list of Animations) to animation manager. """
        # if manager is in instant mode, remove any existing tasks
        if self.instant:
            self.task_list = []

        # add task's objects and their initial position to dict
        if isinstance(task, Animation):
            self._record_start_param_value(task)
        elif isinstance(task, list):
            for animation in task:
                self._record_start_param_value(animation)

        # append task
        self.task_list.append(task)
        self.manager_active = True

    def return_to_start(self, func_handle=ease_out, duration=1):
        """ Animated return of all objects to their global starting position.
            Call after all desired animations have been added. """
        anim_list = []
        for obj, record in self.objects_and_start_params.items():
            for anim_type, start_param in record.items():
                """ Change this! 
                    Better for all animation types to destination as one parameter ex:
                    destination: tuple(x, y) for Move
                    color: tuple(r, g, b, a) for Color
                    angle int for Rotate
                """
                if isinstance(start_param, pyglet.math.Vec2):
                    animation = anim_type(obj, start_param.x, start_param.y, func_handle, duration)
                    anim_list.append(animation)
                elif isinstance(start_param, int) or isinstance(start_param, float):
                    animation = anim_type(obj, start_param, func_handle, duration)
                    anim_list.append(animation)
                elif isinstance(start_param, tuple) and len(start_param) in (3, 4):
                    animation = anim_type(obj, start_param, func_handle, duration)
                    anim_list.append(animation)
        self.task_list.append(anim_list)

    def _record_start_param_value(self, animation_instance: Animation):
        """ Stores references to all objects in task list, their attached animations
            and their global starting parameters (one per animation type). """
        obj = animation_instance.object
        # list of available animations and respective initial parameters
        # should be updated when new animation are added!
        anim_param = {
            Move: pyglet.math.Vec2(obj.x, obj.y),
            Rotate: getattr(obj, "rotation", None),
            Color: getattr(obj, "color", None)
        }

        if obj not in self.objects_and_start_params.keys():
            # add new object to the list
            self.objects_and_start_params[obj] = {}
            for anim_type, param_value in anim_param.items():
                if isinstance(animation_instance, anim_type):
                    self.objects_and_start_params[obj][anim_type] = param_value
        else:
            # object is already in the list
            for anim_type, param_value in anim_param.items():
                if isinstance(animation_instance, anim_type):
                    # if that object doesn't already have assigned parameter for this type
                    if not anim_type in self.objects_and_start_params[obj]:
                        # assign new starting parameter for this animation type
                        self.objects_and_start_params[obj][anim_type] = param_value

    def play(self):
        """ Check if task is running and call its update method. """
        # manager is active if not paused, and have at least one running animation
        if self.manager_active:
            # task is animation or list of animations
            task = self.task_list[self._task_index]
            if isinstance(task, list):
                # this flag is to make sure all animations are finished before moving on
                at_least_one_running = False
                for animation in task:
                    if animation.running:
                        animation.update()
                        at_least_one_running = True
                if not at_least_one_running:
                    if self._mode == 0:
                        self._next_task()
            else:
                if task.running:
                    task.update()
                else:
                    if self._mode == 0:
                        self._next_task()

    def _next_task(self):
        """ Performs checks, then increases index or calls on completion methods. """
        no_of_tasks = len(self.task_list)
        if self._task_index < no_of_tasks - 1:
            self._task_index += 1
        else:
            self.on_completion()
            if self._loop:
                self.reset_all()

    def _prev_task(self):
        """ Perform checks, then decrease index to go back to previous task."""
        if self._task_index > 0:
            self._task_index -= 1

    def _reset_all_tasks(self):
        """ Resets entire play list back to initial conditions in two steps:
            1) re-enables each task by calling its reset() method
            2) moving all objects to positions they had before any animation was played
        """
        for task in self.task_list:
            if isinstance(task, list):
                for animation in task:
                    animation.reset()
            else:
                task.reset()

    def _restore_starting_parameters(self):
        for obj, record in self.objects_and_start_params.items():
            anim_set_command = {
                Move: "position",
                Rotate: "rotation",
                Color: "color",
            }
            for animation_type, start_parameter in record.items():
                for anim, set_command in anim_set_command.items():
                    if anim == animation_type:
                        setattr(obj, set_command, start_parameter)

    def reset_all(self):
        self._task_index = 0
        self._reset_all_tasks()
        self._restore_starting_parameters()

    def reset_task(self, task_index: int):
        task = self.task_list[task_index]
        if isinstance(task, list):
            for animation in task:
                animation.reset()
        else:
            task.reset()

    def complete_task(self, task_index: int):
        task = self.task_list[task_index]
        if isinstance(task, list):
            for animation in task:
                animation.complete()
        else:
            task.complete()

    def pause(self):
        """ Play/pause switch for the animation manager. """
        self.manager_active = not self.manager_active

    def next(self):
        """ Increase task_index to go to next queued task in the task_list. """
        # if task is called before completion complete the first task
        self.complete_task(self._task_index)
        self._next_task()

    def repeat(self):
        self.reset_task(self._task_index)

    def previous(self):
        """ Resets current and previous task, and sets index to previous task. """
        self.reset_task(self._task_index)
        self._prev_task()
        self.reset_task(self._task_index)

    def on_completion(self):
        """ Optional method to run after all animations have finished playing. """
        if self.on_completion_func:
            self.on_completion_func()

    @property
    def loop(self) -> bool:
        """ Returns the state of the loop setting. """
        return self._loop

    @loop.setter
    def loop(self, state: bool):
        self._loop = state

    @property
    def play_mode(self) -> int:
        return self._mode

    @play_mode.setter
    def play_mode(self, mode: int):
        """ Setter for playing mode of the animation manager.
            :param mode: int corresponding to one of following options:
                    0 - automatic continuous animation playing
                    1 - manual sequential playing using "next" and "previous" methods.
                    2 - manual smooth seeking with "forwards" and "backwards" methods.
        """
        if mode in (0, 1, 2):
            self._mode = mode
        else:
            self._mode = 0
            print(f"No playing mode that corresponds to argument {mode}. Mode set to 0.")

    @property
    def current_task(self) -> int:
        """ Returns index of currently active task from the task list."""
        return self._task_index

    @current_task.setter
    def current_task(self, index):
        """ Set active task by passing index of that task in the task list."""
        if index in range(0, len(self.task_list) - 1):
            self._task_index = index
