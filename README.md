
function OnEvent(event, arg)
    if event == "MOUSE_BUTTON_PRESSED"
        and (arg == 4 or arg == 5) then

        OutputLogMessage(
            "Side button=%d, time=%d\n",
            arg,
            GetRunningTime()
        )
    end
end
