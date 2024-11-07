import pyglet
from abc import ABC, abstractmethod
from typing import Callable, Union
from tools.easing import *

FPS = 100

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


class LinearMoveCustom(Animation):
    """ Animates motion of object to given destination with lerp. """

    def __init__(self, obj, dest_x, dest_y, duration=2):
        super().__init__(obj)
        self.dest_x = dest_x
        self.dest_y = dest_y
        self.duration = duration
        self.destination_vector = pyglet.math.Vec2(self.dest_x, self.dest_y)
        self.initial_position_vector = None
        self.delta_s = None
        self.delta_s_mag = 0.0
        self.chunk = None
        self.chunk_mag = 0.0
        self.internal_clock = 0.0
        self.delta_time = 1/FPS
        self.iterations = self.duration / self.delta_time

    def update(self):
        if not self.initial_position_vector:
            self.initial_position_vector = pyglet.math.Vec2(self.object.x, self.object.y)
            self.delta_s = self.destination_vector - self.initial_position_vector
            self.chunk = self.delta_s / (self.iterations, self.iterations)
            self.delta_s_mag = (self.delta_s.x**2 + self.delta_s.y**2) ** 0.5
            self.chunk_mag = (self.chunk.x**2 + self.chunk.y**2) ** 0.5

        self.object.pip_position += self.chunk
        self.delta_s_mag -= self.chunk_mag

        if self.delta_s_mag <= 0:
            self.running = False

class Rotate(Animation):
    def __init__(self, obj, angle, func_handle: Callable = linear, duration=1):
        super().__init__(obj, func_handle, duration)
        #
        if isinstance(obj, pyglet.shapes.Rectangle):
            obj.anchor_x, obj.anchor_y = obj.width/2, obj.height/2
        self.starting_angle = None
        self.current_angle = None
        self.dest_angle = pyglet.math.Vec2(angle, 0)
        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time


    def _set_start_vector(self):
        """
        Records original rotation value, for reset purposes
        """
        if not self.starting_angle:
            self.starting_angle = pyglet.math.Vec2(self.object.rotation, 0)

    def update(self):
        self._set_start_vector()
        if not self.current_angle:
            self.current_angle = pyglet.math.Vec2(self.object.rotation, 0)
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.current_angle = self.current_angle.lerp(self.dest_angle, t)
            self.object.rotation = self.current_angle.x
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self._set_start_vector()  # needed in sequential manager mode
        self.object.rotation -= self.dest_angle.x
        self.current_angle = None

class Color(Animation):
    pass

class Scale(Animation):
    pass

class MoveBACKUP(Animation):
    """
        Moves Sprite or Shape object to destination in given time with Vec2 lerp.
        Uses easing functions from python Arcade for "t" parameter.
    """

    def __init__(self, obj, dest_x, dest_y, func_handle: Callable = linear, duration=1):
        super().__init__(obj, func_handle, duration)
        self.start_vector = pyglet.math.Vec2(self.object.x, self.object.y)
        self.destination_vector = pyglet.math.Vec2(dest_x, dest_y)
        self.position_vector = None
        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time

    def update(self):
        if not self.position_vector:
            self.position_vector = pyglet.math.Vec2(self.object.x, self.object.y)
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.object.pip_position = self.position_vector.lerp(self.destination_vector, t)
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self.object.pip_position = self.start_vector

class Move(Animation):
    """
    Moves pyglet.Sprite or pyglet.Shape object to destination in given time, with Vec2's lerp.
    Lerp parameter "t" here is given as a ratio of elapsed frames to max frames and takes values 0.0 - 1.0.
    Optional easing curve functions are available in easing.py taken from python Arcade.
    """

    def __init__(self, obj, dest_x, dest_y, func_handle: Callable = linear, duration=1):
        """
        :param obj: pyglet.Shape or pyglet.Sprite object to apply animation to
        :param dest_x: x-coordinate of the end point passed object will be moved to
        :param dest_y: y-coordinate of the end point passed object will be moved to
        :param func_handle: Optional easing curve function handle for lerp parameter "t"
        :param duration: desired length of an animation from start to finish in seconds
        """
        super().__init__(obj, func_handle, duration)
        # set destination position vector from passed arguments
        self.destination_vector = pyglet.math.Vec2(dest_x, dest_y)
        # starting position vector is a constant, first set when animation is executed
        self.start_vector = None
        # current position vector is a variable, also set first when animation is executed
        self.position_vector = None
        self.delta_time = 1 / FPS
        self.frames = 0
        self.max_frames = self.duration / self.delta_time

    def _set_start_vector(self):
        """
        Records position of an object when animation first called in Vec2 form.
        If reset is called for this animation, object will be returned to this position.
        """
        if not self.start_vector:
            self.start_vector = pyglet.math.Vec2(self.object.x, self.object.y)

    def update(self):
        if not self.position_vector:
            self.position_vector = pyglet.math.Vec2(self.object.x, self.object.y)
        if self.frames > self.max_frames:
            self.running = False
        else:
            percent = self.frames / self.max_frames
            t = self.function(percent)
            self.object.pip_position = self.position_vector.lerp(self.destination_vector, t)
            self.frames += 1

    def reset(self):
        """ Resets animations attributes, so it can be played again (when looping)"""
        self.frames = 0
        self.running = True
        self._set_start_vector()  # needed in sequential manager mode
        self.object.pip_position = self.start_vector

class AnimationManagerObsolete:
    """
        List that stores and logic that manages animations of objects.

        Animations can be added with add() method and will play in sequence.
        If instead a list of animations is passed to add() method, all will
        play simultaneously.

        Multiple managers can be instantiated to play animation sets independently.
        Instant mode manager interrupts previous animation and plays new one immediately.

        To implement:
        Play, pause, go back? Loop? Play list?
    """

    def __init__(self, on_completion_func=None, instant_mode=False):
        self.on_completion_func = on_completion_func
        self.animation_list = []
        self.manager_active = False
        self.instant = instant_mode

    def add(self, animation: Union[Move, list[Move, ...]]):
        """ Append new animation to animation manager. """
        if self.instant:
            self.animation_list = []
        self.animation_list.append(animation)
        self.manager_active = True

    def play(self):
        """ Call in on_draw to play and manage all added animations. """
        if self.manager_active:
            if not self.animation_list:
                self.manager_active = False
                self.on_completion()
            else:
                animation = self.animation_list[0]
                if isinstance(animation, list):
                    if not animation:
                        # remove empty list
                        self.animation_list.remove(animation)
                    else:
                        for animation_element in animation:
                            self.step(animation_element, animation)
                else:
                    self.step(animation, self.animation_list)

    def on_completion(self):
        """ Run after all animations have finished playing. """
        if self.on_completion_func:
            self.on_completion_func()

    @staticmethod
    def step(animation: Move, animation_list: list):
        """ Run animation's update methods to step animation to next frame. """
        if animation.running:
            animation.update()
        else:
            animation_list.remove(animation)

class AnimationManagerV2Obsolete:
    """
        List that stores and logic that manages animations of objects.

        Animations can be added with add() method and will play in sequence.
        If instead a list of animations is passed to add() method, all will
        play simultaneously.

        Multiple managers can be instantiated to play animation sets independently.
        Instant mode manager interrupts previous animation and plays new one immediately.

        Manager can play in automatic "continuous" mode, or user controlled "sequential" mode.

        To implement:
        Ultimately, some kind of "manual" mode would allow animation to  flow
        forward (and possibly backwards!) smoothly when key is pressed.

        Multiple manager is useful, but requires separate restarting key binding,
        draw calls etc. What if instead multiple task lists are added (under a name?)
        to a same manager? All could be independently played by controlled with single
        controls.

    """

    def __init__(self, on_completion_func=None, instant_mode=False, loop_mode=False):
        self.on_completion_func = on_completion_func
        self.manager_active = False
        self.task_list = []
        self.instant = instant_mode

        self._task_index = 0
        self._loop = loop_mode

    def add(self, animation: Union[Animation, list[Animation, ...]]):
        """ Append new task (Animation or list of Animations) to animation manager."""
        if self.instant:
            self.task_list = []
        self.task_list.append(animation)
        self.manager_active = True

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
                    self._next_task()
            else:
                if task.running:
                    task.update()
                else:
                    self._next_task()

    def _next_task(self):
        """ Performs checks, then increases index or calls on completion methods. """
        no_of_tasks = len(self.task_list)
        if self._task_index < no_of_tasks - 1:
            self._task_index += 1
        else:
            self.on_completion()
            if self._loop:
                self.reset()
            else:
                self.manager_active = False

    def pause(self):
        """ Play/pause switch for the animation manager. """
        self.manager_active = not self.manager_active

    def reset(self):
        """ Resets all animations. """
        self._task_index = 0
        for task in self.task_list:
            if isinstance(task, list):
                for animation in task:
                    animation.reset()
            else:
                task.reset()

    def next(self):
        """ Increase task_index to go to next queued task in the task_list. """
        raise NotImplementedError

    def previous(self):
        """ Decrease task_index to return to previous task in the task_list. """
        raise NotImplementedError

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, state: bool):
        self._loop = state

    def on_completion(self):
        """ Optional method to run after all animations have finished playing. """
        if self.on_completion_func:
            self.on_completion_func()

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

        To implement:
        recheck Animaiton/Move class and initial position restoring
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
        self.objects_and_origins = {}
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
            # if isinstance(task, Rotate):
            #     obj = task.object
            #     if obj not in self.objects_and_origins.keys():
            #         self.objects_and_origins[obj] = obj.rotation
            # else:
            obj = task.object
            if obj not in self.objects_and_origins.keys():
                self.objects_and_origins[obj] = obj.x, obj.y
        elif isinstance(task, list):
            for animation in task:
                obj = animation.object
                if obj not in self.objects_and_origins.keys():
                    self.objects_and_origins[obj] = obj.x, obj.y

        # append task
        self.task_list.append(task)
        self.manager_active = True

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

    def pause(self):
        """ Play/pause switch for the animation manager. """
        self.manager_active = not self.manager_active

    def _reset_all_tasks(self):
        """
            Resets entire play list back to initial conditions in two steps:
            1) re-enables each task by calling its reset() method
            2) moving all objects to positions they had before any animation was played
        """
        for task in self.task_list:
            if isinstance(task, list):
                for animation in task:
                    animation.reset()
            else:
                task.reset()

    def _restore_initial_positions(self):
        for obj, pos in self.objects_and_origins.items():
            obj.pip_position = pos

    def reset_all(self):
        self._task_index = 0
        self._reset_all_tasks()
        self._restore_initial_positions()

    def reset_task(self, task: Animation):
        """ Reset passed task by calling reset() for all its elements."""
        if isinstance(task, list):
            for animation in task:
                animation.reset()
        else:
            task.reset()

    def next(self):
        """ Increase task_index to go to next queued task in the task_list. """
        self._next_task()
        self.reset_task(self.task_list[self._task_index])

    def repeat(self):
        self.reset_task(self.task_list[self._task_index])

    def previous(self):
        """ Decrease task_index to return to previous task in the task_list. """
        self.reset_task(self.task_list[self._task_index])
        self._prev_task()
        self.reset_task(self.task_list[self._task_index])

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
        """
        Setter for playing mode of the animation manager.
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


