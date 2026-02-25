import matplotlib
matplotlib.use("Agg")   # ← 禁用 Tkinter GUI，避免 mainloop error

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq


# ============================================================
# Choose which color route to compute
# ============================================================

SHOW_COLOR = "all"     # "orange", "green", ... 或 "all"


# ============================================================
# HSV color definitions
# ============================================================

COLOR_BASE = {
    "yellow": ((18, 170, 160), (80, 255, 255)),
    "orange": ((0, 160, 101), (17, 255, 255)),
    "purple": ((121, 40, 0), (160, 255, 217)),
    "green":  ((45, 99, 102), (93, 255, 255)),
    "mint":   ((60, 24, 175), (93, 71, 255)),
    "brown":  ((13, 97, 111), (23, 120, 255))
}


# ============================================================
# HSV buffer
# ============================================================

def expand_hsv(low, high, h=5, s=20, v=30):
    low_new = (max(0, low[0] - h),
               max(0, low[1] - s),
               max(0, low[2] - v))

    high_new = (min(179, high[0] + h),
                min(255, high[1] + s),
                min(255, high[2] + v))

    return low_new, high_new


# ============================================================
# Mask cleaning
# ============================================================

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m


# ============================================================
# A* Search
# ============================================================

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, goal, nodes, max_step=300):
    pq = []
    heapq.heappush(pq, (0, start))

    came_from = {}
    g = {tuple(start): 0}

    nodes = [tuple(n) for n in nodes]

    while pq:
        _, current = heapq.heappop(pq)
        current = tuple(current)

        if current == tuple(goal):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for nxt in nodes:
            if nxt == current:
                continue

            d = dist(current, nxt)
            if d > max_step:
                continue

            new_cost = g[current] + d

            if nxt not in g or new_cost < g[nxt]:
                g[nxt] = new_cost
                priority = new_cost + dist(nxt, goal)
                heapq.heappush(pq, (priority, nxt))
                came_from[nxt] = current

    return None


# ============================================================
# Hold detection
# ============================================================

def detect_holds(mask, threshold=300):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holds = []

    for cnt in cnts:
        if cv2.contourArea(cnt) < threshold:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        holds.append((cnt, cx, cy))

    return holds


# ============================================================
# Overlay detection drawing
# ============================================================

def overlay_smart(img, holds):
    out = img.copy()

    for cnt, cx, cy in holds:
        cv2.drawContours(out, [cnt], -1, (0,0,255), 2)

        cnt = cnt[:,0,:]
        num_points = len(cnt)
        if num_points < 10:
            continue

        cv2.circle(out, (cx, cy), 6, (255,255,0), -1)

        hull = cv2.convexHull(cnt, returnPoints=False)
        has_good = False

        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt.reshape(-1,1,2), hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, depth = defects[i,0]
                    depth /= 256.0
                    far = tuple(cnt[f])

                    if depth > 4 and far[1] < cy:
                        has_good = True

                        if s < e:
                            seg = cnt[s:e+1]
                        else:
                            seg = np.vstack((cnt[s:], cnt[:e+1]))

                        cv2.polylines(out, [seg.reshape(-1,1,2)], False, (0,255,0), 3)
                        cv2.circle(out, far, 4, (255,0,0), -1)

        if not has_good:
            top_idx = np.argmin(cnt[:,1])
            rng = max(int(num_points * 0.08), 5)
            idxs = [(top_idx + i) % num_points for i in range(-rng, rng+1)]
            seg = cnt[idxs].reshape(-1,1,2)
            cv2.polylines(out, [seg], False, (0,255,0), 3)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ============================================================
# MAIN
# ============================================================

def main():
    img = cv2.imread(r"C:\Code\Computer_vision\Project\wall.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for name, (lo, hi) in COLOR_BASE.items():

        if SHOW_COLOR != "all" and name != SHOW_COLOR:
            continue

        if name == "yellow":
            lo_b, hi_b = lo, hi
        else:
            lo_b, hi_b = expand_hsv(lo, hi)

        mask = cv2.inRange(hsv, np.array(lo_b), np.array(hi_b))
        mask = clean_mask(mask)

        # detect holds
        holds_raw = detect_holds(mask)
        nodes = [(cx, cy) for (_, cx, cy) in holds_raw]

        # A*
        route = None
        start = goal = None

        if len(nodes) >= 2:
            start = min(nodes, key=lambda p: p[1])
            goal  = max(nodes, key=lambda p: p[1])
            route = a_star(start, goal, nodes)

        # base overlay
        overlay = cv2.cvtColor(overlay_smart(img, holds_raw), cv2.COLOR_RGB2BGR)

        # draw route
        if route:
            for i in range(len(route)-1):
                p1 = tuple(map(int, route[i]))
                p2 = tuple(map(int, route[i+1]))
                cv2.line(overlay, p1, p2, (0,255,255), 6)

            cv2.circle(overlay, tuple(map(int, start)), 12, (0,255,255), -1)
            cv2.circle(overlay, tuple(map(int, goal)), 12, (0,255,255), -1)

        # save output
        save_path = fr"C:\Code\Computer_vision\Project\output_route_{name}.png"

        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print("Saved route:", save_path)


if __name__ == "__main__":
    main()
