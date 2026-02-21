#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

struct MBOEvent {
    int action = 0;        // Add=0, Cancel=1, Modify=2, Trade=3
    float price = 0.0f;
    uint32_t size = 0;
    int side = 0;          // Bid=0, Ask=1
    uint64_t ts_event = 0;
};

class DayEventBuffer {
public:
    DayEventBuffer() = default;

    void load(const std::string& dbn_path, uint32_t instrument_id) {
        // Real implementation would parse .dbn.zst file.
        // For unit tests, this is a no-op placeholder.
        (void)dbn_path;
        (void)instrument_id;
    }

    std::span<const MBOEvent> get_events(uint32_t begin, uint32_t end) const {
        if (begin >= end || begin >= events_.size()) {
            return {};
        }
        uint32_t actual_end = std::min(end, static_cast<uint32_t>(events_.size()));
        return std::span<const MBOEvent>(events_.data() + begin, actual_end - begin);
    }

    size_t size() const { return events_.size(); }

    void clear() { events_.clear(); events_.shrink_to_fit(); }

private:
    std::vector<MBOEvent> events_;
};
