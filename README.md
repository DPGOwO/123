
function OnEvent(event, arg)
    if event == "MOUSE_BUTTON_PRESSED" and arg == 4 then

        OutputLogMessage("Click START\n")

        PressMouseButton(1)
        Sleep(100)
        ReleaseMouseButton(1)

        OutputLogMessage("Click END\n")
    end
end
