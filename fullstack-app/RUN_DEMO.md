# Final Presentation Demo Startup Guide

This document contains the EXACT commands to run the project successfully on a fresh boot.

## Important Notes Before Starting
- Ensure your laptop is connected to the presentation Wi-Fi network.
- Determine your laptop's current LAN IP (e.g. 192.168.x.x) and use it when necessary.

---

### Terminal 1: Core Backend (Express + Prisma + Neon)
**Directory**: `simmplificant-open-updated/services/core-backend`
**Command**: 
```bash
npm run dev
```
**Expected Output**:
```
Connected to PostgreSQL (development)
SIH26090 core-backend listening on http://localhost:4000
```
*(If you see EADDRINUSE for 4000, find the process and kill it)*

### Terminal 2: FastAPI AI Service
**Directory**: `image-enhanced-updated`
**Command**:
```bash
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:visual_forge:Loading rembg model...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Terminal 3: Web Frontend (Vite)
**Directory**: `simmplificant-open-updated`
**Command**:
```bash
npm run web:dev
```
**Expected Output**:
```
VITE v6.4.3  ready in XXX ms
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://<YOUR_LAN_IP>:5173/
```
*Note down the Network LAN IP!*

### Terminal 4: Mobile App (Expo)
**Directory**: `simmplificant-open-updated`

**Command**:
```bash
npm run mobile:start
```
**Expected Output**:
```
Starting project at ...
Starting Metro Bundler
Waiting on http://localhost:8081
```

---

## Phone Setup Steps (NO USB/ADB REQUIRED)
1. Make sure your physical phone is on the **exact same Wi-Fi network** as your laptop.
2. Open the **Expo Go** app manually on your Android device.
3. Tap **"Scan QR Code"** and scan the code shown in Terminal 4 (press `c` to show it if needed).
4. **DO NOT press `a` in the terminal** if you don't have a configured USB debugging ADB setup. Pressing `a` attempts to reverse-proxy over USB, which causes the `adb.exe: device offline` error if the phone is not properly authorized.
5. The application will bundle and launch the WebView over LAN.
6. Grant Camera and Microphone permissions when prompted.
7. The WebView will point to the local Vite dev server.

## Troubleshooting Common Failures
* **Mobile App Shows Blank/Error Screen**: 
  - Ensure the Vite app LAN IP is reachable from your phone's browser. If it is not, check your Windows Firewall settings.
  - Verify the hardcoded IP in `apps/mobile/App.tsx` matches the Vite Network IP.
* **Microphone Does Not Record**: 
  - In Expo Go, long-press to reload, ensuring permissions are granted. The web app uses the native bridge to request audio recording from the shell instead of the browser `getUserMedia` to prevent permission issues.
* **Product Publish Fails**: 
  - Check Terminal 1 logs. Make sure the `.env` `DATABASE_URL` is pointing to Neon and you are successfully authenticated.
* **AI Enhance Fails**: 
  - Check Terminal 2 logs. Ensure `.env` contains the required Groq and Cloudinary credentials.

## Testing & Verification
All core demo flows have been tested:
- ✅ Express API connects to Neon PostgreSQL
- ✅ FastAPI starts and handles Cloudinary enhancement and Groq audio cataloging
- ✅ Web application properly proxies `/api/v1` and `/api/ai`
- ✅ Artisan Login (JWT auth) generates and stores correctly
- ✅ Mobile WebView bridge injects native recording hooks
- ✅ Marketplace properly surfaces published products
- ✅ Buyer ordering correctly updates stock and handles 409 Insufficient Stock
