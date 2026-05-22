import pygame
import numpy as np
import numpy.random as random
from Model import Model


""""
Simple code to visualize the running of the model
Made with help from Claude
"""

# ── colours ────────────────────────────────────────────────────────────────
BG          = (15,  15,  20)
PANEL_BG    = (22,  22,  30)
ACCENT      = (0,  210, 180)
BTN_GREEN   = (30, 160,  90)
BTN_ORANGE  = (210, 110,  30)
BTN_BLUE    = (40, 120, 200)
GRAPH_BG    = (18,  18,  25)
TEXT        = (210, 220, 230)
TEXT_DIM    = (110, 120, 135)
DIVIDER     = (40,  42,  55)

# FIX #7: cap history length to prevent unbounded list growth
MAX_HIST = 500


# ── tiny UI widgets ─────────────────────────────────────────────────────────
class Slider:
    H = 6

    def __init__(self, label, x, y, w, lo, hi, init, fmt=str, callback=None):
        self.label    = label
        self.track    = pygame.Rect(x, y + 18, w, self.H)
        self.lo, self.hi = lo, hi
        self.value    = init
        self.fmt      = fmt
        self.callback = callback
        self.dragging = False

    @property
    def _handle_x(self):
        t = (self.value - self.lo) / (self.hi - self.lo)
        return int(self.track.x + t * self.track.w)

    def draw(self, surf, font):
        # FIX #2: compute handle x once, no walrus operator needed
        hx = self._handle_x
        # track
        pygame.draw.rect(surf, (50, 52, 65), self.track, border_radius=3)
        # filled portion
        filled = pygame.Rect(self.track.x, self.track.y,
                             hx - self.track.x, self.H)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        # handle
        pygame.draw.circle(surf, (230, 235, 245), (hx, self.track.centery), 8)
        pygame.draw.circle(surf, ACCENT,           (hx, self.track.centery), 5)
        # label
        surf.blit(font.render(f"{self.label}:  {self.fmt(self.value)}",
                              True, TEXT), (self.track.x, self.track.y - 18))

    def handle(self, event, offset):
        """offset = (panel_screen_x, panel_screen_y)"""
        hx = self._handle_x + offset[0]
        hy = self.track.centery + offset[1]
        if event.type == pygame.MOUSEBUTTONDOWN:
            if abs(event.pos[0]-hx) < 12 and abs(event.pos[1]-hy) < 12:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            t = (event.pos[0] - offset[0] - self.track.x) / self.track.w
            self.value = self.lo + t * (self.hi - self.lo)
            self.value = max(self.lo, min(self.hi, self.value))
            if isinstance(self.lo, int):
                self.value = int(round(self.value))
            if self.callback:
                self.callback(self.value)


class Button:
    def __init__(self, label, x, y, w, h, callback, color=BTN_GREEN):
        self.label    = label
        self.rect     = pygame.Rect(x, y, w, h)
        self.color    = color
        self.callback = callback
        self._hover   = False

    def draw(self, surf, font):
        col = tuple(min(255, c + 25) for c in self.color) if self._hover else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        t = font.render(self.label, True, (255, 255, 255))
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle(self, event, offset):
        sr = self.rect.move(offset)
        if event.type == pygame.MOUSEMOTION:
            self._hover = sr.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and sr.collidepoint(event.pos):
            self.callback()


# ── main window ─────────────────────────────────────────────────────────────
class Window:
    PANEL_W   = 240
    GRAPH_H   = 180       # pixels reserved at the bottom for the graph
    FPS       = 60

    def __init__(self, win_w=1100, win_h=680):
        pygame.init()
        self.screen  = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Evolution of Evolvable GRN")
        self.clock   = pygame.time.Clock()

        self.W, self.H = win_w, win_h
        self.grid_w  = win_w - self.PANEL_W
        self.grid_h  = win_h - self.GRAPH_H

        self.model     = None
        self.auto_run  = False
        self.sim_speed = 50          # steps / second
        self._auto_acc = 0.0        # ms accumulator

        self.min_hist  = []
        self.mean_hist = []

        self.font   = pygame.font.SysFont("consolas", 13)
        self.font_b = pygame.font.SysFont("consolas", 13, bold=True)

        # pre-allocated surfaces
        self._grid_surf  = pygame.Surface((self.grid_w, self.grid_h))
        self._graph_surf = pygame.Surface((self.grid_w, self.GRAPH_H))
        self._panel_surf = pygame.Surface((self.PANEL_W, self.H))

        self._build_ui()
        # FIX #1: __init__ calls _reset once; _build_ui must run first so
        # sliders exist, but the model is None at that point — _reset handles it.
        self._reset()

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        px, pw = 18, self.PANEL_W - 36

        # Re-spaced vertically (y = 25, 80, 135, 190) to leave room for the new slider
        self.sliders = [
            Slider("Fitness Power", px, 25, pw, 1, 20, 10,
                   callback=lambda v: self.model and setattr(self.model, 'FitnessPower', v)),
            Slider("Death Rate", px, 80, pw, 1, 500, 100,
                   fmt=lambda v: f"{v / 1000:.3f}",
                   callback=lambda v: self.model and setattr(self.model, 'DeathRate', v / 1000)),
            # NEW: Float slider from 0.0 to 5.0 (Passing floats prevents integer snapping)
            Slider("Mut. Factor", px, 135, pw, 1.0, 25.0, 1.0,
                   fmt=lambda v: f"{v:.2f}",
                   callback=lambda v: self.model and setattr(self.model, 'MutationFactor', v)),
            Slider("Sim Speed", px, 190, pw, 1, 500, 50,
                   callback=lambda v: setattr(self, 'sim_speed', v)),
        ]

        bh, gap = 34, 6
        by = 255  # Shifted down slightly to clear the fourth slider cleanly
        self.buttons = [
            Button("Reset", px, by, pw, bh, self._reset, BTN_GREEN),
            Button("Step", px, by + bh + gap, pw, bh, self._step, BTN_GREEN),
            Button("▶  Auto Run", px, by + 2 * (bh + gap), pw, bh, self._toggle_auto, BTN_GREEN),
            Button("Switch Target", px, by + 3 * (bh + gap), pw, bh, self._switch, BTN_BLUE),
        ]
        self._auto_btn = self.buttons[2]

    # ── model actions ────────────────────────────────────────────────────────
    # FIX #1: single, unified _reset that is safe to call at any time
    def _reset(self):
        fp = self.sliders[0].value if hasattr(self, 'sliders') else 10
        dr = (self.sliders[1].value / 1000) if hasattr(self, 'sliders') else 0.01
        # NEW: Capture the current value of your new mutation factor slider
        mf = self.sliders[2].value if hasattr(self, 'sliders') else 1.0

        # Pass it straight to your updated Model instantiation call
        self.model = Model(death_rate=dr, fitness_power=fp, mutation_factor=mf)
        self.min_hist.clear()
        self.mean_hist.clear()
        self._auto_acc = 0.0

    def _step(self):
        if self.model is None:
            return
        self.model.ExecuteStep()
        self.min_hist.append(self.model.MinimalDistance)
        self.mean_hist.append(self.model.MeanDistance)
        # FIX #7: cap history so graph line-drawing stays O(MAX_HIST)
        if len(self.min_hist) > MAX_HIST:
            self.min_hist  = self.min_hist[-MAX_HIST:]
            self.mean_hist = self.mean_hist[-MAX_HIST:]

    def _toggle_auto(self):
        self.auto_run = not self.auto_run
        if self.auto_run:
            self._auto_btn.label = "⏸  Pause"
            self._auto_btn.color = BTN_ORANGE
        else:
            self._auto_btn.label = "▶  Auto Run"
            self._auto_btn.color = BTN_GREEN

    def _switch(self):
        if self.model:
            self.model.SwitchTarget()

    # ── fast grid render via surfarray ───────────────────────────────────────
    def _render_grid(self):
        surf = self._grid_surf
        surf.fill(BG)

        if self.model is None or self.model.Grid is None:
            msg = self.font.render("No model loaded", True, TEXT_DIM)
            surf.blit(msg, msg.get_rect(center=(self.grid_w // 2, self.grid_h // 2)))
            return surf

        xs, ys = self.model.xSize, self.model.ySize
        sw, sh = self.grid_w, self.grid_h

        # FIX #3: three states — dead (-1), stable alive (fitness)
        fitness  = np.full((xs, ys), -1.0, dtype=np.float32)

        # FIX #6: still requires a Python loop over object array; mitigate by
        # keeping the body as lean as possible (no attribute chaining).
        grid = self.model.Grid
        for i in range(xs):
            for j in range(ys):
                c = grid[i, j]
                if c is None:
                    continue
                if c.IsStable:
                    fitness[i, j] = float(c.Fitness)

        # Map fitness -> RGB
        stable_mask   = fitness >= 0
        f             = np.clip(fitness, 0, 1)
        r = np.where(stable_mask,  ((1 - f) * 255).astype(np.uint8), 0)
        g = np.where(stable_mask,  (f        * 255).astype(np.uint8), 0)
        b = np.zeros_like(r)


        rgb = np.stack([r, g, b], axis=2)   # (xs, ys, 3)

        # FIX #10: use pygame.transform.scale to fill the area exactly
        cell_px = max(1, min(sw // xs, sh // ys))
        scaled  = np.repeat(np.repeat(rgb, cell_px, axis=0), cell_px, axis=1)
        small   = pygame.surfarray.make_surface(scaled)
        # Scale to fill grid area precisely, eliminating edge clipping
        pygame.transform.scale(small, (sw, sh), surf)
        return surf

    # ── graph ────────────────────────────────────────────────────────────────
    def _render_graph(self):
        surf = self._graph_surf
        surf.fill(GRAPH_BG)

        pad = 36
        gw  = self.grid_w - 2 * pad
        gh  = self.GRAPH_H - 2 * pad
        max_val = 20

        # axes
        pygame.draw.line(surf, DIVIDER, (pad, pad),    (pad,    pad+gh))
        pygame.draw.line(surf, DIVIDER, (pad, pad+gh), (pad+gw, pad+gh))

        # y ticks
        for v in range(0, 21, 5):
            y = pad + gh - int(v / max_val * gh)
            pygame.draw.line(surf, DIVIDER, (pad-4, y), (pad+gw, y))
            surf.blit(self.font.render(str(v), True, TEXT_DIM), (2, y-7))

        n = len(self.min_hist)
        if n < 2:
            surf.blit(self.font.render("Waiting for data…", True, TEXT_DIM),
                      (pad+8, pad + gh//2))
            return surf

        def px(i, v):
            x = pad + int(i / (n-1) * gw)
            y = pad + gh - int(v / max_val * gh)
            return x, y

        # draw lines
        for i in range(n-1):
            pygame.draw.line(surf, (0, 210, 220),
                             px(i, self.min_hist[i]),  px(i+1, self.min_hist[i+1]),  2)
            pygame.draw.line(surf, (255, 160, 40),
                             px(i, self.mean_hist[i]), px(i+1, self.mean_hist[i+1]), 2)

        # legend
        pygame.draw.line(surf, (0,210,220),   (pad, 8), (pad+20, 8), 2)
        surf.blit(self.font.render("min dist",  True, (0,210,220)),   (pad+24, 1))
        pygame.draw.line(surf, (255,160,40),  (pad+90, 8), (pad+110, 8), 2)
        surf.blit(self.font.render("mean dist", True, (255,160,40)), (pad+114, 1))

        # FIX #7: show how many steps are displayed if history was capped
        if len(self.min_hist) == MAX_HIST:
            surf.blit(self.font.render(f"(last {MAX_HIST} steps)", True, TEXT_DIM),
                      (pad+200, 1))

        return surf

    # ── side panel ───────────────────────────────────────────────────────────
    def _render_panel(self):
        surf = self._panel_surf
        surf.fill(PANEL_BG)

        # header
        hdr = self.font_b.render("GRN  EVOLUTION", True, ACCENT)
        surf.blit(hdr, hdr.get_rect(centerx=self.PANEL_W//2, y=8))
        pygame.draw.line(surf, ACCENT, (12, 26), (self.PANEL_W-12, 26))

        for s in self.sliders:
            s.draw(surf, self.font)
        for b in self.buttons:
            b.draw(surf, self.font)

        # stats block
        sy = 460
        pygame.draw.line(surf, DIVIDER, (12, sy-8), (self.PANEL_W-12, sy-8))

        # FIX #9: explicit "no model" message when model is None
        if self.model is None:
            surf.blit(self.font.render("No model loaded", True, TEXT_DIM), (18, sy))
        else:
            stats = [
                ("Timestep",   getattr(self.model, 'Timestep', 0)),
                ("Population", getattr(self.model, 'TotalPopulation', '—')),
                ("Min dist",   getattr(self.model, 'MinimalDistance', '—')),
                ("Mean dist",  f"{getattr(self.model,'MeanDistance',0):.2f}"),
                ("Target",     "A" if np.array_equal(
                                   getattr(self.model,'Target', None),
                                   getattr(self.model,'TargetA', None)) else "B"),
            ]
            for label, val in stats:
                surf.blit(self.font.render(f"{label:<12} {val}", True, TEXT), (18, sy))
                sy += 20
        return surf

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        panel_origin = (self.grid_w, 0)

        while True:
            dt = self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                for s in self.sliders:
                    s.handle(event, panel_origin)
                for b in self.buttons:
                    b.handle(event, panel_origin)

            if self.auto_run and self.model:
                self._auto_acc += dt
                interval = 1000 / max(1, self.sim_speed)
                # FIX #8: clamp accumulator to one interval to prevent step bursts
                # (e.g. after a speed increase or window focus regain)
                self._auto_acc = min(self._auto_acc, interval * 2)
                while self._auto_acc >= interval:
                    self._step()
                    self._auto_acc -= interval

            # draw
            self.screen.fill(BG)
            self.screen.blit(self._render_grid(),  (0, 0))
            self.screen.blit(self._render_graph(), (0, self.grid_h))
            self.screen.blit(self._render_panel(), panel_origin)

            # dividers
            pygame.draw.line(self.screen, DIVIDER,
                             (0, self.grid_h), (self.grid_w, self.grid_h))
            pygame.draw.line(self.screen, DIVIDER,
                             (self.grid_w, 0), (self.grid_w, self.H))

            pygame.display.flip()


if __name__ == "__main__":
    Window().run()