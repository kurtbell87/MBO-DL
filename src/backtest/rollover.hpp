#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ContractSpec — quarterly contract definition
// ---------------------------------------------------------------------------
struct ContractSpec {
    std::string symbol;
    uint32_t instrument_id = 0;
    int start_date = 0;
    int end_date = 0;
    int rollover_date = 0;
};

// ---------------------------------------------------------------------------
// RolloverCalendar — manages contract transitions
// ---------------------------------------------------------------------------
class RolloverCalendar {
public:
    RolloverCalendar() = default;

    void add_contract(const ContractSpec& spec) {
        contracts_.push_back(spec);
    }

    const std::vector<ContractSpec>& contracts() const { return contracts_; }

    // Excluded dates: rollover date + 3 days before each rollover
    bool is_excluded(int date) const {
        for (const auto& c : contracts_) {
            if (date == c.rollover_date) return true;
            // 3 days before rollover
            if (date >= c.rollover_date - 3 && date < c.rollover_date) return true;
        }
        return false;
    }

    std::optional<ContractSpec> get_contract_for_date(int date) const {
        for (const auto& c : contracts_) {
            if (date >= c.start_date && date <= c.end_date) {
                return c;
            }
        }
        return std::nullopt;
    }

    std::set<int> excluded_dates() const {
        std::set<int> dates;
        for (const auto& c : contracts_) {
            dates.insert(c.rollover_date);
            dates.insert(c.rollover_date - 1);
            dates.insert(c.rollover_date - 2);
            dates.insert(c.rollover_date - 3);
        }
        return dates;
    }

private:
    std::vector<ContractSpec> contracts_;
};
