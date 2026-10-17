# DI-04 Chimney Census | Simple System Architecture

> **The Big Idea**: Think of this project as a **smart speed-camera in space** that helps environmental officers find illegal, polluting brick factories across thousands of square kilometers in seconds.

---

## 1. The Story Behind the System

Imagine you are **Inspector Farida Khan**:
* You have **1 jeep**, **2 staff members**, and a territory of **3,000 square kilometers** (the size of an entire district).
* Hidden in rural fields are hundreds of brick kilns.
* Old kilns (**FCBK**) create heavy black smoke and are now illegal. Newer kilns (**Zigzag**) are cleaner and legal.
* **The Problem**: You cannot physically drive down every dirt road to check which kilns are legal and which are polluting. It would take years and cost too much fuel.
* **Our Solution**: We use European satellites orbiting 500 km above Earth and an onboard **AI brain** to spot every kiln, figure out its design, and hand Farida a prioritized **Top 20 inspection list** every morning.

---

## 2. The 4-Step Journey: From Space to the Jeep

```
   [ 🛰️ STEP 1: SATELLITE CAMERA ]
   European Sentinel-2 satellite flies 500 km in space.
   Takes pictures of the district (at 10 meters per pixel).
                  │
                  ▼
   [ 🧠 STEP 2: THE AI BRAIN (YOLO-OBB) ]
   The AI looks at the tiny 12-pixel smudge.
   It instantly figures out:
   • Is it an illegal Oval kiln (FCBK)?
   • Is it a clean Rectangular kiln (Zigzag)?
   • It rotates a box tightly around the kiln.
                  │
                  ▼
   [ 🗺️ STEP 3: THE SMART MAP CLEANER ]
   When satellite photos overlap, the computer makes sure
   no factory is counted twice.
   It finds kilns that sit close together (within 1 km).
                  │
                  ▼
   [ 📋 STEP 4: INSPECTOR FARIDA'S ACTION PLAN ]
   Instead of thousands of messy photos, Farida gets:
   1. A 2-page printout with a clear map.
   2. The Top 20 worst polluters to visit today.
   3. Grouped routes so she can inspect 3 kilns in 1 trip.
```

---

## 3. What Each Part Does (In Simple Words)

| Part of the System | What It Does (Non-Tech Translation) | Real-Life Analogy |
| :--- | :--- | :--- |
| **Sentinel-2 Satellite** | Takes free photos of the district every few days from orbit. | A high-altitude drone that never runs out of battery. |
| **AI Detector (YOLO-OBB)** | Scans the photo in **2 milliseconds** and recognizes the shape of the kiln. | Facial recognition, but for industrial chimneys. |
| **Slanted Bounding Box (OBB)** | Draws a rotated rectangle that tilts with the kiln instead of a square. | Tailoring a suit to fit your body instead of wearing a cardboard box. |
| **CubeSat Edge Chip (RPi 5)** | Runs the AI directly inside the satellite while still in space. | Checking your passport at the gate instead of shipping your paperwork across the globe. |
| **Downlink Telemetry** | Sends only a tiny text alert down to Earth instead of giant images. | Texting "Found 2 illegal kilns at Lat/Lon" instead of emailing a 4K movie. |
| **Route Grouping (1 km)** | Finds kilns sitting near each other. | Delivery apps grouping food orders on the same street. |
| **The 2-Page Brief** | A short, clean paper summary ready for the field. | A doctor's 1-page prescription instead of 500 pages of medical books. |

---

## 4. The 3 "Secret Sauces" That Make This System Great

### 1. Slanted Boxes (Why We Don't Use Standard Rectangles)
Brick kilns sit at all kinds of diagonal angles in farm fields.
* **Standard AI**: Draws a wide, straight box that captures 70% farm dirt, trees, and mud, confusing the computer.
* **Our AI**: Rotates the box neatly around the kiln like a glove, ignoring background dirt.

### 2. Space AI (Why We Put the Brain Onboard)
Usually, satellites send giant photo files down to ground stations, which costs thousands of dollars and takes hours or days.
* **Our System**: The AI runs **onboard the satellite** on a tiny computer chip (using only 4.2 Watts—less than a nightlight). It drops a tiny **1.2 KB text message** down to Earth in seconds.

### 3. Smart Fuel-Saving Routes
Inspector Farida only has one jeep. Our system finds clusters where 2 or 3 kilns operate within walking distance of each other. She can park once and inspect multiple sites, saving **up to 65% in patrol fuel and time**.

---

## 5. Summary: What the Final User Sees

When an environmental officer opens the system:
1. **No coding**: A simple dark-mode dashboard with clear red (illegal), green (clean), and orange (heritage) pins.
2. **One-Click Download**: A 2-page PDF ready to print on the clipboard before driving out.
3. **Zero Guesswork**: The GPS coordinates, factory type, and reason for inspection are already written out.
