# Quick Start Guide - Accessibility Vision Assistant

## 🚀 5-Minute Setup

### Step 1: Install Python
Download Python 3.8+ from https://www.python.org/downloads/

### Step 2: Install the Project

Open PowerShell in the project folder and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# Run automated setup
python setup.py
```

### Step 3: Get API Key (Optional but Recommended)

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Open `.env` file
5. Replace `your_api_key_here` with your actual key

### Step 4: Run the Application

```powershell
python app.py
```

Select option `1` for GUI mode. Your browser will open automatically!

## 🎯 Quick Feature Test

### Test Object Detection
1. Click "webcam" in the interface
2. Select "🎯 Object Detection"
3. Click "🔍 Analyze"
4. Listen to the audio feedback!

### Test Text Reading
1. Upload an image with text (or use webcam)
2. Select "📄 Text Reading (OCR)"
3. Click "🔍 Analyze"
4. Hear the text read aloud!

### Test AI Assistant
1. Go to "💬 AI Assistant" tab
2. Upload an image
3. Type a question like "What do you see?"
4. Get an intelligent response!

## ❓ Common First-Run Issues

### "ModuleNotFoundError"
**Solution**: Make sure you activated the virtual environment
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### "Camera not found"
**Solution**: 
- Check webcam is connected
- Grant camera permissions when prompted
- Try restarting the app

### "CUDA not available" (This is OK!)
**Note**: The app is designed to run on CPU. This warning is normal.

### Slow on first run
**Note**: First run downloads models (~100MB). This is normal and only happens once.

## 🎓 Usage Tips

### For Best Audio Experience
- Use headphones or good speakers
- Keep audio feedback enabled
- Adjust system volume to comfortable level

### For Best Vision Accuracy
- Good lighting is important
- Hold objects 1-2 feet from camera
- Keep camera steady for text reading
- Closer is better for barcodes

### For Low-End PCs
- Use only one feature at a time
- Close other applications
- Select lower quality webcam settings if available
- Avoid "Complete Scene Analysis" if too slow

## 📱 Features Overview

| Feature | Use Case | Speed |
|---------|----------|-------|
| 🎯 Object Detection | "What's around me?" | ⚡⚡⚡ Fast |
| 👤 Face Detection | "How many people?" | ⚡⚡⚡ Fast |
| 📱 QR/Barcode | Scan products | ⚡⚡⚡ Fast |
| 📄 Text Reading | Read signs/labels | ⚡⚡ Medium |
| 🎨 Color ID | "What color is this?" | ⚡⚡⚡ Fast |
| 🤖 AI Scene | Ask questions | ⚡ Slower (needs API) |
| 🔍 Complete Analysis | Full scene understanding | ⚡ Slower |

## 🔧 Keyboard Shortcuts

- `Enter` - Analyze image
- `Tab` - Navigate between controls
- `Space` - Toggle checkboxes
- Click interface elements for full accessibility

## 📞 Need Help?

Check `README.md` for:
- Detailed feature descriptions
- Troubleshooting guide
- Advanced configuration
- Technical details

## ✅ Quick Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] .env file created with API key (optional)
- [ ] Webcam connected and working
- [ ] Audio output working
- [ ] Ran `python app.py` successfully

## 🎉 You're Ready!

Enjoy using the Accessibility Vision Assistant. This tool is designed to make visual information accessible through technology.

---

**Questions?** Check the full README.md or CHANGELOG.md for more details.
