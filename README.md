-- 左键按下时间（ms）
local PRESS_MIN  = 100
local PRESS_MAX  = 180
local PRESS_MEAN = 140
local PRESS_STD  = 20

-- 左键松开时间（ms）
local RELEASE_MIN  = 10
local RELEASE_MAX  = 25
local RELEASE_MEAN = 17.5
local RELEASE_STD  = 4



local pressTime = boundedGaussian(
    PRESS_MEAN, PRESS_STD,
    PRESS_MIN, PRESS_MAX
)

local releaseTime = boundedGaussian(
    RELEASE_MEAN, RELEASE_STD,
    RELEASE_MIN, RELEASE_MAX
)

PressMouseButton(1)

local ok = waitOrStop(pressTime)

ReleaseMouseButton(1)

if not ok then
    break
end

if not waitOrStop(releaseTime) then
    break
end
