"""A utility module that simplifies pymunk integration with pyglet
TODO: A Body blueprint, a circle body with update, remove and constraints
"""
import math
import random

import pyglet.shapes
import pymunk


class PhysicsBody:
    pass


class CircleBody(PhysicsBody):
    def __init__(
            self,
            space: pymunk.Space,
            batch: pyglet.graphics.Batch,
            radius=10,
            mass=1,
            position=(0, 0),
            velocity=(0, 0),
            friction=0.5,
            elasticity=0.75,
            color=(255, 255, 255, 255),
            body_type=pymunk.Body.DYNAMIC
    ):
        # body parameters
        self.space = space
        self.radius = radius
        self.mass = mass
        self.type = body_type
        self.moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(self.mass, self.moment, self.type)
        self.body.position = position
        self.body.velocity = velocity
        # shape parameters
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity

        self.space.add(self.body, self.shape)

        # pyglet shape parameters
        self.batch = batch
        self.color = color
        self.visual = pyglet.shapes.Circle(
            self.body.position.x, self.body.position.y, self.radius, color=self.color, batch=self.batch
        )

    def delete(self):
        """Removes elements from pymunk space and rendering batch."""
        self.space.remove(self.body, self.shape)
        self.visual.delete()

    def update(self):
        """Align position and orientation of visual representation and simulated body."""
        self.visual.position = self.body.position

    @property
    def position(self):
        return self.body.position

    @position.setter
    def position(self, value: tuple[float, float]):
        """Change position of the simulated body, and its visual representation. Doesn't work for STATIC."""
        self.body.position = value
        self.update()

    @property
    def velocity(self):
        return self.body.velocity

    @velocity.setter
    def velocity(self, value: tuple[float, float]):
        """Change velocity of the simulated body."""
        self.body.velocity = value


class RectangleBody(PhysicsBody):
    def __init__(
            self,
            space: pymunk.Space,
            batch: pyglet.graphics.Batch,
            width=100,
            height=100,
            mass=1,
            position=(0, 0),
            velocity=(0, 0),
            friction=0.5,
            elasticity=0.75,
            color=(255, 255, 255, 255),
            body_type=pymunk.Body.DYNAMIC
    ):
        # body parameters
        self.space = space
        self.width = width
        self.height = height
        self.mass = mass
        self.type = body_type
        self.moment = pymunk.moment_for_box(self.mass, (self.width, self.height))
        self.body = pymunk.Body(self.mass, self.moment, self.type)
        self.body.position = position
        self.body.velocity = velocity
        # shape parameters
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.friction = friction
        self.shape.elasticity = elasticity

        self.space.add(self.body, self.shape)

        # pyglet shape parameters
        self.batch = batch
        self.color = color
        self.visual = pyglet.shapes.Rectangle(
            self.body.position.x,
            self.body.position.y,
            self.width,
            self.height,
            color=self.color[:3],
            batch=self.batch
        )
        self.visual.anchor_position = self.width/2, self.height/2

    def delete(self):
        self.space.remove(self.body, self.shape)
        self.visual.delete()

    def update(self):
        self.visual.position = self.body.position
        self.visual.rotation = -math.degrees(self.body.angle)

    @property
    def position(self):
        return self.body.position

    @position.setter
    def position(self, value: tuple[float, float]):
        """Change position of the simulated body, and its visual representation. Doesn't work for STATIC."""
        self.body.position = value
        self.update()

    @property
    def velocity(self):
        return self.body.velocity

    @velocity.setter
    def velocity(self, value: tuple[float, float]):
        """Change velocity of the simulated body."""
        self.body.velocity = value


