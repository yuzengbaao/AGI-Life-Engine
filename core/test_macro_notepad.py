Sub NotepadAutomationMacro
    ' 声明变量
    Dim WshShell
    Set WshShell = CreateObject("WScript.Shell")
    
    ' 打开记事本
    WshShell.Run "notepad.exe"
    WScript.Sleep 1000  ' 等待记事本启动
    
    ' 输入指定文本
    WshShell.AppActivate "无标题 - 记事本"
    WScript.Sleep 500
    WshShell.SendKeys "这是一段由UI宏自动生成的测试文本。"
    WScript.Sleep 500
    
    ' 保存文件为 output_test_macro.txt
    WshShell.SendKeys "^s"  ' Ctrl+S 打开另存为对话框
    WScript.Sleep 800
    
    WshShell.SendKeys "output_test_macro.txt"
    WScript.Sleep 300
    
    WshShell.SendKeys "%s"  ' Alt+S 保存文件
    WScript.Sleep 500
    
    ' 可选：关闭记事本（如果需要）
    ' WshShell.SendKeys "^{F4}"
    
    ' 清理对象
    Set WshShell = Nothing
    
    WScript.Echo "宏执行完成：文件已保存为 output_test_macro.txt"
End Sub