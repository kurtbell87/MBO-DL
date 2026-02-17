#pragma once

#include "bars/bar.hpp"
#include "bars/volume_bar_builder.hpp"
#include "bars/tick_bar_builder.hpp"
#include "bars/dollar_bar_builder.hpp"
#include "bars/time_bar_builder.hpp"

#include <memory>
#include <string>

class BarFactory {
public:
    static std::unique_ptr<BarBuilder> create(const std::string& type, double threshold) {
        if (type == "volume") {
            return std::make_unique<VolumeBarBuilder>(static_cast<uint32_t>(threshold));
        } else if (type == "tick") {
            return std::make_unique<TickBarBuilder>(static_cast<uint32_t>(threshold));
        } else if (type == "dollar") {
            return std::make_unique<DollarBarBuilder>(threshold);
        } else if (type == "time") {
            return std::make_unique<TimeBarBuilder>(static_cast<uint64_t>(threshold));
        }
        return nullptr;
    }
};
