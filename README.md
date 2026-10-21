
-- Logitech G102 / G HUB
-- 随机间隔左键连点宏
-- G4 按一次开启，再按一次关闭

-- ===== 用户配置 =====
local TOGGLE_BUTTON = 4

-- 点击周期：两次按下左键之间的时间（ms）
local INTERVAL_MIN  = 110
local INTERVAL_MAX  = 170
local INTERVAL_MEAN = 135
local INTERVAL_STD  = 15

-- 每次左键保持按下的时间（ms）
local PRESS_MIN  = 25
local PRESS_MAX  = 45
local PRESS_MEAN = 34
local PRESS_STD  = 5

-- ===== 内部状态 =====
local running = false
local ignoreUntil = 0
local seeded = false

-- Box-Muller：生成标准正态分布随机数
local function gaussian()
    local u1 = math.max(math.random(), 0.000001)
    local u2 = math.random()

    return math.sqrt(-2 * math.log(u1))
        * math.cos(2 * math.pi * u2)
end

-- 截断正态分布，保证结果不超出边界
local function boundedGaussian(mean, std, minVal, maxVal)
    for i = 1, 20 do
        local value = math.floor(
            mean + gaussian() * std + 0.5
        )

        if value >= minVal and value <= maxVal then
            return value
        end
    end

    -- 极少数情况下采样失败，回退到合法均值
    return math.max(minVal, math.min(maxVal, mean))
end

-- 分段等待，同时检测关闭侧键
local function waitOrStop(ms)
    local elapsed = 0

    while elapsed < ms do
        if IsMouseButtonPressed(TOGGLE_BUTTON) then
            return false
        end

        local step = math.min(5, ms - elapsed)
        Sleep(step)
        elapsed = elapsed + step
    end

    return not IsMouseButtonPressed(TOGGLE_BUTTON)
end

function OnEvent(event, arg)

    if event ~= "MOUSE_BUTTON_PRESSED"
        or arg ~= TOGGLE_BUTTON then
        return
    end

    if running or GetRunningTime() < ignoreUntil then
        return
    end

    if not seeded then
        math.randomseed(GetRunningTime() + 10007)
        math.random()
        math.random()
        seeded = true
    end

    running = true
    OutputLogMessage("Rapid Fire ON\n")

    -- 等待开启侧键松开
    while IsMouseButtonPressed(TOGGLE_BUTTON) do
        Sleep(10)
    end

    Sleep(50)

    while running do
        if IsMouseButtonPressed(TOGGLE_BUTTON) then
            break
        end

        -- 随机生成完整点击周期
        local interval = boundedGaussian(
            INTERVAL_MEAN,
            INTERVAL_STD,
            INTERVAL_MIN,
            INTERVAL_MAX
        )

        -- 随机生成按下时长
        local pressTime = boundedGaussian(
            PRESS_MEAN,
            PRESS_STD,
            PRESS_MIN,
            PRESS_MAX
        )

        -- 保证按下时长小于周期
        pressTime = math.min(pressTime, interval - 1)

        PressMouseButton(1)

        local continue = waitOrStop(pressTime)

        ReleaseMouseButton(1)

        if not continue then
            break
        end

        -- 剩余时间保持左键松开
        if not waitOrStop(interval - pressTime) then
            break
        end
    end

    ReleaseMouseButton(1)

    while IsMouseButtonPressed(TOGGLE_BUTTON) do
        Sleep(10)
    end

    running = false
    ignoreUntil = GetRunningTime() + 300

    OutputLogMessage("Rapid Fire OFF\n")
end
