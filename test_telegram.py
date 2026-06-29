"""Test script to verify Telegram notifications."""

from src.utils.notifier import TelegramNotifier

notifier = TelegramNotifier()

test_analytics = {
    "period_days": 30,
    "total_bets": 50,
    "won_bets": 28,
    "lost_bets": 22,
    "hit_rate": 56.0,
    "yield_pct": 8.5,
    "roi_pct": 8.5,
    "total_pnl": 425.50,
    "total_staked": 5000.00,
    "avg_clv": 0.0234,
    "avg_odds": 1.85,
}

print("Sending test message to Telegram...")
success = notifier.send_performance_report(test_analytics)

if success:
    print("✓ Message sent successfully!")
else:
    print("✗ Failed to send message")
