# Windows Installation Guide

## Prerequisites Check

Before starting, verify you have:
- [ ] Windows 10 or 11
- [ ] Webcam connected
- [ ] Microphone working
- [ ] Internet connection
- [ ] ~2GB free disk space

## Step-by-Step Installation

### 1. Install Python

**Download Python:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 (recommended) or 3.8+
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Click "Install Now"

**Verify Installation:**
```powershell
python --version
```
Should show: `Python 3.11.x` or similar

### 2. Open Project in PowerShell

```powershell
# Navigate to project folder
cd "C:\Users\jagat\Downloads\smart ai\smart-visual-assistant"
```

### 3. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# You should see (venv) in your prompt
```

### 4. Upgrade Pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

**Option A: Automated (Recommended)**
```powershell
python setup.py
```

**Option B: Manual**
```powershell
pip install -r requirements.txt
```

⏳ This will take 5-10 minutes. Downloads ~500MB of packages.

### 6. Setup Configuration

```powershell
# Copy example env file
copy .env.example .env

# Edit with Notepad
notepad .env
```

Add your Gemini API key:
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Paste in .env file replacing `your_api_key_here`
5. Save and close

### 7. Run the Application

```powershell
python app.py
```

Select option `1` for GUI mode. Browser will open automatically!

---

## Common Windows Issues & Fixes

### Issue: "python is not recognized"

**Solution:**
1. Reinstall Python
2. **IMPORTANT**: Check "Add Python to PATH"
3. Restart PowerShell

### Issue: "Execution Policy" Error

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Microsoft Visual C++ required"

**Solution:**
Download and install from:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue: pip install fails with "error: Microsoft Visual C++ 14.0 is required"

**Solution for specific packages:**

For `pyzbar`:
```powershell
pip install pipwin
pipwin install pyzbar
```

For general C++ requirements:
Install Visual Studio Build Tools from:
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

### Issue: Camera not working

**Solution:**
1. Settings → Privacy → Camera
2. Allow apps to access camera
3. Allow desktop apps to access camera

### Issue: Microphone not working

**Solution:**
1. Settings → Privacy → Microphone
2. Allow apps to access microphone
3. Test in Sound settings

### Issue: "Module not found" errors

**Solution:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Slow performance

**Solutions:**
1. Close other applications
2. Use only one feature at a time
3. Lower webcam resolution (edit code if needed)
4. Disable Windows visual effects:
   - System → Advanced system settings → Performance → Adjust for best performance

---

## Verify Installation

Run these commands to check everything works:

```powershell
# Test Python
python -c "print('Python OK')"

# Test imports
python -c "import cv2; print('OpenCV OK')"
python -c "import gradio; print('Gradio OK')"
python -c "import pyttsx3; print('TTS OK')"
```

All should print "OK" without errors.

---

## Running After Installation

**Every time you want to use the app:**

```powershell
# 1. Navigate to project
cd "C:\Users\jagat\Downloads\smart ai\smart-visual-assistant"

# 2. Activate virtual environment
.\venv\Scripts\activate

# 3. Run app
python app.py
```

---

## Creating a Desktop Shortcut

Create a file `run_assistant.bat` with:

```batch
@echo off
cd /d "C:\Users\jagat\Downloads\smart ai\smart-visual-assistant"
call venv\Scripts\activate.bat
python app.py
pause
```

Right-click → Create shortcut → Drag to Desktop

---

## Uninstall

To completely remove:

```powershell
# 1. Delete virtual environment
Remove-Item -Recurse -Force venv

# 2. Optionally delete entire project folder
Remove-Item -Recurse -Force "C:\Users\jagat\Downloads\smart ai\smart-visual-assistant"
```

---

## Performance Tips for Windows

1. **Disable Windows Search indexing** for project folder
2. **Add exception in Windows Defender** for project folder
3. **Close background apps** (especially Chrome, Discord)
4. **Use high performance power plan**
5. **Keep Windows updated**

---

## Getting Help

If installation fails:

1. Check this guide for your specific error
2. Read `README.md` for detailed info
3. Check `PROJECT_SUMMARY.md` for technical details
4. Ensure all prerequisites are met

---

## Success Checklist

- [ ] Python installed and in PATH
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] .env file created with API key
- [ ] `python app.py` runs without errors
- [ ] GUI opens in browser
- [ ] Camera works in interface
- [ ] Audio feedback working

✅ If all checked, you're ready to use the Accessibility Vision Assistant!
