#pragma once

class WarmupTracker {
public:
    WarmupTracker() = default;

    bool is_warmup(int bar_index, int ewma_span = 20) const {
        return bar_index < ewma_span;
    }
};
