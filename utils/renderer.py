import pickle
import numpy as np
import pygame
import gymnasium as gym


class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: gym.Env
    zoom: int = 4

    def __init__(self, envs, model):
        self.envs = envs
        self.env = envs.venv.envs[0]
        self.model = model
        self.current_frame = self.env._obj_obs
        self._init_pygame(self.current_frame)
        self.paused = False

        self.current_keys_down = set()
        self.current_mouse_pos = None
        self.keys2actions = self.env.oc_env.unwrapped.get_keys_to_action()

        self.ram_grid_anchor_left = self.env_render_shape[0] + 28
        self.ram_grid_anchor_top = 28

        self.active_cell_idx = None
        self.candidate_cell_ids = []
        self.current_active_cell_input : str = ""

        self.rgb_agent = False
        self.human_playing = False

    def _init_pygame(self, sample_image):
        pygame.init()
        pygame.display.set_caption("OCAtari Environment")
        sample_image = np.repeat(np.repeat(np.swapaxes(sample_image, 0, 1), self.zoom, axis=0), self.zoom, axis=1)
        print(sample_image.shape)
        self.env_render_shape = sample_image.shape[:2]
        window_size = self.env_render_shape[:2]
        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        

    def run(self):
        self.running = True
        while self.running:
            self._handle_user_input()
            if not self.paused:
                if self.human_playing:
                    action = self._get_action()
                elif self.rgb_agent:
                    pass
                else:
                    action, _ = self.model.predict(self.env.sco_obs, deterministic=True)
                    print(action)
                tuple = self.env.step(action)
                self.env.sco_obs = tuple[0]
                rew = tuple[1]
                self.current_frame = self.env._obj_obs
            self._render()
        pygame.quit()

    def _get_action(self):
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()
                
                elif event.key == pygame.K_h:  # 'H': toggle human playing
                    self.human_playing = not self.human_playing
                    if self.human_playing:
                        print("Human playing")
                    else:
                        print("AI playing")
                
                elif event.key == pygame.K_m:  # 'M': save snapshot
                    snapshot = self.env._ale.cloneState()
                    pickle.dump(snapshot, open("snapshot.pkl", "wb"))
                    print("Saved snapshot.pkl")

                elif event.key == pygame.K_ESCAPE and self.active_cell_idx is not None:
                    self._unselect_active_cell()

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

                elif pygame.K_0 <= event.key <= pygame.K_9:  # enter digit
                    char = str(event.key - pygame.K_0)
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input += char

                elif event.key == pygame.K_BACKSPACE:  # remove character
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input = self.current_active_cell_input[:-1]

                elif event.key == pygame.K_RETURN:
                    if self.active_cell_idx is not None:
                        if len(self.current_active_cell_input) > 0:
                            new_cell_value = int(self.current_active_cell_input)
                            if new_cell_value < 256:
                                self._set_ram_value_at(self.active_cell_idx, new_cell_value)
                        self._unselect_active_cell()

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

    def _render(self, frame = None):
        self.window.fill((0,0,0))  # clear the entire window
        self._render_atari(frame)
        pygame.display.flip()
        pygame.event.pump()

    def _render_atari(self, frame = None):
        if frame is None:
            frame = self.current_frame
        frame = np.swapaxes(np.repeat(np.repeat(frame, self.zoom, axis=0), self.zoom, axis=1), 0, 1)
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))
        self.clock.tick(60)