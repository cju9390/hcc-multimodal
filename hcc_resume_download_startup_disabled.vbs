Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c cd /d D:\hcc_multimodal && ""C:\Users\FORYOUCOM\AppData\Local\Python\pythoncore-3.14-64\python.exe"" -u scripts\02_download_pilot.py >> download_185.log 2>&1", 0, False
