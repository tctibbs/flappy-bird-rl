"""Player (bird) entity."""

from enum import Enum
from itertools import cycle
from typing import TYPE_CHECKING

import pygame

from reinforced_flapper.game.entities.entity import Entity
from reinforced_flapper.game.entities.floor import Floor
from reinforced_flapper.game.entities.pipe import Pipe, Pipes
from reinforced_flapper.game.utils import clamp

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class PlayerMode(Enum):
    """Possible player modes."""

    SHM = "SHM"
    NORMAL = "NORMAL"
    CRASH = "CRASH"


class Player(Entity):
    """The bird.

    Attributes:
        mode: Current player mode.
        min_y: Minimum y position.
        max_y: Maximum y position.
        crashed: True once the player has collided.
        crash_entity: What the player collided with ("pipe" or "floor").
    """

    def __init__(self, config: "GameConfig") -> None:
        """Initialize the player.

        Args:
            config: Shared game context.
        """
        image = config.images.player[0]
        x = int(config.window.width * 0.2)
        y = int((config.window.height - image.get_height()) / 2)
        super().__init__(config, image, x, y)
        self.min_y = -2 * self.h
        self.max_y = config.window.viewport_height - self.h * 0.75
        self.img_idx = 0
        self.img_gen = cycle([0, 1, 2, 1])
        self.frame = 0
        self.crashed = False
        self.crash_entity: str | None = None
        self.set_mode(PlayerMode.SHM)

    def set_mode(self, mode: PlayerMode) -> None:
        """Switch the player mode and reset mode-specific physics.

        Args:
            mode: The mode to switch to.
        """
        self.mode = mode
        if mode == PlayerMode.NORMAL:
            self.reset_vals_normal()
            self.config.sounds.wing.play()
        elif mode == PlayerMode.SHM:
            self.reset_vals_shm()
        elif mode == PlayerMode.CRASH:
            self.stop_wings()
            self.config.sounds.hit.play()
            if self.crash_entity == "pipe":
                self.config.sounds.die.play()
            self.reset_vals_crash()

    def reset_vals_normal(self) -> None:
        """Reset physics values for normal play."""
        self.vel_y = -9  # velocity along Y axis
        self.max_vel_y = 10  # max descend speed
        self.min_vel_y = -8  # max ascend speed
        self.acc_y = 1  # downward acceleration

        self.rot = 80  # current rotation
        self.vel_rot = -3  # rotation speed
        self.rot_min = -90  # min rotation angle
        self.rot_max = 20  # max rotation angle

        self.flap_acc = -9  # velocity gained on flap
        self.flapped = False  # True for the frame after a flap

    def reset_vals_shm(self) -> None:
        """Reset physics values for splash-screen bobbing."""
        self.vel_y = 1
        self.max_vel_y = 4
        self.min_vel_y = -4
        self.acc_y = 0.5

        self.rot = 0
        self.vel_rot = 0
        self.rot_min = 0
        self.rot_max = 0

        self.flap_acc = 0
        self.flapped = False

    def reset_vals_crash(self) -> None:
        """Reset physics values for the crash fall."""
        self.acc_y = 2
        self.vel_y = 7
        self.max_vel_y = 15
        self.vel_rot = -8

    def tick_shm(self) -> None:
        """Update position in SHM mode."""
        if self.vel_y >= self.max_vel_y or self.vel_y <= self.min_vel_y:
            self.acc_y *= -1
        self.vel_y += self.acc_y
        self.y += self.vel_y

    def tick_normal(self) -> None:
        """Update position in normal mode."""
        if self.vel_y < self.max_vel_y and not self.flapped:
            self.vel_y += self.acc_y
        if self.flapped:
            self.flapped = False

        self.y = clamp(self.y + self.vel_y, self.min_y, self.max_y)
        self.rotate()

    def tick_crash(self) -> None:
        """Update position in crash mode."""
        if self.min_y <= self.y <= self.max_y:
            self.y = clamp(self.y + self.vel_y, self.min_y, self.max_y)
            # rotate only when it's a pipe crash and bird is still falling
            if self.crash_entity != "floor":
                self.rotate()

        if self.vel_y < self.max_vel_y:
            self.vel_y += self.acc_y

    def rotate(self) -> None:
        """Advance the rotation angle."""
        self.rot = clamp(self.rot + self.vel_rot, self.rot_min, self.rot_max)

    def tick(self) -> None:
        """Update the player for one frame."""
        if self.mode == PlayerMode.SHM:
            self.tick_shm()
        elif self.mode == PlayerMode.NORMAL:
            self.tick_normal()
        elif self.mode == PlayerMode.CRASH:
            self.tick_crash()

    def render(self) -> None:
        """Draw the player."""
        self.update_animation()
        self.draw_player()

    def update_animation(self) -> None:
        """Advance the wing-flap animation."""
        self.frame += 1
        if self.frame % 5 == 0:
            self.img_idx = next(self.img_gen)
            self.image = self.config.images.player[self.img_idx]
            self.w = self.image.get_width()
            self.h = self.image.get_height()

    def draw_player(self) -> None:
        """Blit the rotated player sprite."""
        if self.image is None:
            return
        rotated_image = pygame.transform.rotate(self.image, self.rot)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        self.config.require_screen().blit(rotated_image, rotated_rect)

    def stop_wings(self) -> None:
        """Freeze the wing animation."""
        self.img_gen = cycle([self.img_idx])

    def flap(self) -> None:
        """Flap upward."""
        if self.y > self.min_y:
            self.vel_y = self.flap_acc
            self.flapped = True
            self.rot = 80
            self.config.sounds.wing.play()

    def crossed(self, pipe: Pipe) -> bool:
        """Check whether the player crossed a pipe this frame.

        Args:
            pipe: The pipe to test.

        Returns:
            True if the player's center passed the pipe's center this frame.
        """
        return pipe.cx <= self.cx < pipe.cx - pipe.vel_x

    def collided(self, pipes: Pipes, floor: Floor) -> bool:
        """Check collision against the floor and all pipes.

        Args:
            pipes: The pipe set.
            floor: The floor.

        Returns:
            True if the player collided with anything.
        """
        if self.collide(floor):
            self.crashed = True
            self.crash_entity = "floor"
            return True

        for pipe in pipes.upper:
            if self.collide(pipe):
                self.crashed = True
                self.crash_entity = "pipe"
                return True
        for pipe in pipes.lower:
            if self.collide(pipe):
                self.crashed = True
                self.crash_entity = "pipe"
                return True

        return False
