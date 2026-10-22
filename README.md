while running do
    if IsMouseButtonPressed(TOGGLE_BUTTON) then
        break
    end

    -- 随机生成按下时长
    local pressTime = boundedGaussian(
        PRESS_MEAN,
        PRESS_STD,
        PRESS_MIN,
        PRESS_MAX
    )

    -- 随机生成松开时长
    local releaseTime = boundedGaussian(
        RELEASE_MEAN,
        RELEASE_STD,
        RELEASE_MIN,
        RELEASE_MAX
    )

    -- 按下左键
    PressMouseButton(1)

    local ok = waitOrStop(pressTime)

    -- 松开左键
    ReleaseMouseButton(1)

    if not ok then
        break
    end

    -- 等待下一次点击
    if not waitOrStop(releaseTime) then
        break
    end
end
