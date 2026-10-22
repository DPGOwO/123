
-- CF 加特林-炼狱速点
-- 鼠标 G5 按住启动，松开停止
-- 游戏内需要将开枪键绑定为空格

local lianyu = 5

-- 按键持续时间
local PRESS_MIN = 100
local PRESS_MAX = 185

-- 松开后的间隔时间
local RELEASE_MIN = 10
local RELEASE_MAX = 25

function OnEvent(event, arg)

    if event == "MOUSE_BUTTON_PRESSED"
        and arg == lianyu then

        OutputLogMessage("Lianyu START\n")

        repeat
            -- 模拟空格按下
            PressKey("spacebar")
            Sleep(math.random(PRESS_MIN, PRESS_MAX))

            -- 模拟空格松开
            ReleaseKey("spacebar")
            Sleep(math.random(RELEASE_MIN, RELEASE_MAX))

        until not IsMouseButtonPressed(lianyu)

        -- 确保停止后按键被释放
        ReleaseKey("spacebar")

        OutputLogMessage("Lianyu STOP\n")
    end
end
