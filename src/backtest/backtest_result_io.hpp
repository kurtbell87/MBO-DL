#pragma once

#include "backtest/multi_day_runner.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace backtest_io {

// Escape a string for JSON output
inline std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            default:   result += c;
        }
    }
    return result;
}

inline std::string exit_reason_str(int reason) {
    switch (reason) {
        case exit_reason::TARGET:      return "TARGET";
        case exit_reason::STOP:        return "STOP";
        case exit_reason::TAKE_PROFIT: return "TAKE_PROFIT";
        case exit_reason::EXPIRY:      return "EXPIRY";
        case exit_reason::SESSION_END: return "SESSION_END";
        case exit_reason::SAFETY_CAP:  return "SAFETY_CAP";
        default: return "UNKNOWN";
    }
}

inline std::string label_method_str(OracleConfig::LabelMethod m) {
    switch (m) {
        case OracleConfig::LabelMethod::FIRST_TO_HIT:   return "FIRST_TO_HIT";
        case OracleConfig::LabelMethod::TRIPLE_BARRIER:  return "TRIPLE_BARRIER";
        default: return "UNKNOWN";
    }
}

// Serialize a BacktestResult to JSON
inline std::string to_json(const BacktestResult& result) {
    std::ostringstream ss;
    ss << "{";
    ss << "\"total_trades\":" << result.total_trades;
    ss << ",\"winning_trades\":" << result.winning_trades;
    ss << ",\"losing_trades\":" << result.losing_trades;
    ss << ",\"win_rate\":" << result.win_rate;
    ss << ",\"gross_pnl\":" << result.gross_pnl;
    ss << ",\"net_pnl\":" << result.net_pnl;
    ss << ",\"profit_factor\":" << result.profit_factor;
    ss << ",\"expectancy\":" << result.expectancy;
    ss << ",\"max_drawdown\":" << result.max_drawdown;
    ss << ",\"sharpe\":" << result.sharpe;
    ss << ",\"safety_cap_fraction\":" << result.safety_cap_fraction;

    // Trades array
    ss << ",\"trades\":[";
    for (size_t i = 0; i < result.trades.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& t = result.trades[i];
        ss << "{";
        ss << "\"entry_price\":" << t.entry_price;
        ss << ",\"exit_price\":" << t.exit_price;
        ss << ",\"direction\":" << t.direction;
        ss << ",\"gross_pnl\":" << t.gross_pnl;
        ss << ",\"net_pnl\":" << t.net_pnl;
        ss << ",\"exit_reason\":\"" << exit_reason_str(t.exit_reason) << "\"";
        ss << ",\"bars_held\":" << t.bars_held;
        ss << "}";
    }
    ss << "]";

    ss << "}";
    return ss.str();
}

// Serialize a BacktestResult with config metadata
inline std::string to_json(const BacktestResult& result, const BacktestConfig& cfg) {
    std::ostringstream ss;
    ss << "{";
    ss << "\"bar_type\":\"" << json_escape(cfg.bar_type) << "\"";
    ss << ",\"bar_param\":" << cfg.bar_param;
    ss << ",\"label_method\":\"" << label_method_str(cfg.oracle.label_method) << "\"";
    ss << ",\"target_ticks\":" << cfg.oracle.target_ticks;
    ss << ",\"stop_ticks\":" << cfg.oracle.stop_ticks;

    // Inline the result fields
    ss << ",\"total_trades\":" << result.total_trades;
    ss << ",\"net_pnl\":" << result.net_pnl;
    ss << ",\"expectancy\":" << result.expectancy;

    ss << ",\"trades\":[";
    for (size_t i = 0; i < result.trades.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& t = result.trades[i];
        ss << "{";
        ss << "\"entry_price\":" << t.entry_price;
        ss << ",\"exit_price\":" << t.exit_price;
        ss << ",\"exit_reason\":\"" << exit_reason_str(t.exit_reason) << "\"";
        ss << "}";
    }
    ss << "]";

    ss << "}";
    return ss.str();
}

// Serialize a vector of DayResults
inline std::string to_json(const std::vector<DayResult>& days) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < days.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& d = days[i];
        ss << "{";
        ss << "\"date\":" << d.date;
        ss << ",\"skipped\":" << (d.skipped ? "true" : "false");
        ss << ",\"total_trades\":" << d.result.total_trades;
        ss << ",\"net_pnl\":" << d.result.net_pnl;
        ss << "}";
    }
    ss << "]";
    return ss.str();
}

}  // namespace backtest_io
