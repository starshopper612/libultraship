#ifdef __SWITCH__
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <switch.h>

namespace Ship {

// Small libnx-backed state cache for each logical port used by mapping classes.
enum class GyroSource : uint8_t {
    Left = 0,
    Right = 1,
    Both = 2,
};

struct NXControllerState {
    PadState State = {};
    HidVibrationDeviceHandle Handles[2][2] = {};
    HidSixAxisSensorHandle Sensors[10] = {};  // 6-9: Lucia, Lagon, Lark, Lager (third-party controllers)
    // Style whose vibration init into Handles[1] succeeded.
    uint64_t LastExternalRumbleStyle = 0;
    // Vibration init into Handles[0] succeeded.
    bool HandheldRumbleReady = false;
    // SendRumble runs every game tick; these de-duplicate its logging.
    uint64_t LastInitFailLog[2] = { ~0ULL, ~0ULL };
    int LastSendPath = -1;
    bool LastSendOn = false;
    Result LastSendResult[2] = {};
    int SendLogCount = 0;
    bool Initialized = false;
    GyroSource Gyro = GyroSource::Left;
};

class SwitchController {
  public:
    static SwitchController& GetInstance();
    bool ReadGyro(uint8_t portIndex, float& pitch, float& yaw, float& roll);
    void SetGyroSource(uint8_t portIndex, GyroSource source);
    GyroSource GetGyroSourceForPort(uint8_t portIndex);
    bool IsDetachableJoyConPair(uint8_t portIndex);
    void SendRumble(uint8_t portIndex, float lowFrequencyAmplitude, float highFrequencyAmplitude);
    bool IsNpadConnected(uint8_t portIndex) const;
    std::string GetControllerName(uint8_t portIndex);
    std::string GetControllerSerial(uint8_t npadIndex);

    static void RegisterDevice(int32_t instanceId, int32_t slot, const std::string& serial);
    static void ClearDeviceSlots();
    static int32_t GetDeviceSlot(int32_t instanceId);
    static std::string GetDeviceSerial(int32_t instanceId);

    static void Update();

  private:
    SwitchController() = default;
    bool EnsureInitialized(uint8_t portIndex);
    HidNpadIdType GetNpadId(uint8_t portIndex) const;
    bool ReadSixAxisState(uint8_t portIndex, HidSixAxisSensorState& state);
    bool InitVibration(uint8_t portIndex, int slot, HidNpadIdType npadId, uint64_t style);

    std::array<NXControllerState, 8> mControllers;
};
} // namespace Ship
#endif