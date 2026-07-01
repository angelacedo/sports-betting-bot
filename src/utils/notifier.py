"""
Telegram Notifier Module
Sends betting notifications and performance reports via Telegram Bot API.
Supports Markdown formatting for rich messages.
"""

from typing import Any

import requests

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.utils.logger import logger


class TelegramNotifier:
    """Sends notifications via Telegram Bot API."""

    def __init__(
        self,
        bot_token: str = TELEGRAM_BOT_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID,
    ) -> None:
        """Initialize notifier with bot credentials."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a text message to Telegram chat.

        Args:
            text: Message text (supports Markdown)
            parse_mode: Parse mode (Markdown, HTML, MarkdownV2)

        Returns:
            True if message sent successfully
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Telegram message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_bet_notification(
        self,
        match: str,
        market: str,
        selection: str,
        odds: float,
        stake: float,
        ev: float,
    ) -> bool:
        """
        Send notification for a new bet placed.

        Args:
            match: Match description (e.g., "Arsenal vs Chelsea")
            market: Market type (e.g., "1X2", "Over/Under")
            selection: Bet selection (e.g., "Home", "Over 2.5")
            odds: Decimal odds
            stake: Bet stake
            ev: Expected value

        Returns:
            True if sent successfully
        """
        message = (
            f"🎯 *Nueva Apuesta*\n\n"
            f"📊 Partido: {match}\n"
            f"🎲 Mercado: {market}\n"
            f"✅ Selección: {selection}\n"
            f"💰 Cuota: {odds:.2f}\n"
            f"💵 Stake: ${stake:.2f}\n"
            f"📈 EV: {ev:+.2%}\n\n"
            f"_Apuesta registrada en el sistema_"
        )

        return self.send_message(message)

    def send_performance_report(self, analytics: dict[str, Any]) -> bool:
        """
        Send performance analytics report.

        Args:
            analytics: Dict with performance metrics

        Returns:
            True if sent successfully
        """
        if not analytics or analytics.get("total_bets", 0) == 0:
            message = (
                f"📊 *Reporte de Rendimiento*\n\n"
                f"No hay apuestas en los últimos {analytics.get('period_days', 30)} días."
            )
            return self.send_message(message)

        period = analytics.get("period_days", 30)
        total_bets = analytics.get("total_bets", 0)
        won_bets = analytics.get("won_bets", 0)
        lost_bets = analytics.get("lost_bets", 0)
        hit_rate = analytics.get("hit_rate", 0.0)
        yield_pct = analytics.get("yield_pct", 0.0)
        total_pnl = analytics.get("total_pnl", 0.0)
        total_staked = analytics.get("total_staked", 0.0)
        avg_clv = analytics.get("avg_clv", 0.0)
        avg_odds = analytics.get("avg_odds", 0.0)

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        yield_emoji = "✅" if yield_pct >= 0 else "❌"

        message = (
            f"📊 *Reporte de Rendimiento* ({period} días)\n\n"
            f"🎯 *Apuestas Totales:* {total_bets}\n"
            f"   ✅ Ganadas: {won_bets}\n"
            f"   ❌ Perdidas: {lost_bets}\n\n"
            f"🎯 *Hit Rate:* {hit_rate:.1f}%\n\n"
            f"{yield_emoji} *Yield/ROI:* {yield_pct:+.2f}%\n"
            f"{pnl_emoji} *P&L Total:* ${total_pnl:+.2f}\n"
            f"💵 *Total Apostado:* ${total_staked:.2f}\n\n"
            f"📈 *CLV Promedio:* {avg_clv:+.4f}\n"
            f"💰 *Cuota Promedio:* {avg_odds:.2f}\n\n"
            f"_Generado automáticamente por Sports Betting Bot_"
        )

        return self.send_message(message)

    def send_value_bets(self, value_bets: list[Any], fixtures_analyzed: int) -> bool:
        """
        Send value bets notification.

        Args:
            value_bets: List of ValueBet objects
            fixtures_analyzed: Number of fixtures analyzed

        Returns:
            True if sent successfully
        """
        if not value_bets:
            return self.send_message(
                f"📊 *Análisis Diario*\n\n"
                f"Se analizaron {fixtures_analyzed} partidos.\n"
                f"No se encontraron value bets hoy."
            )

        message = f"🎯 *Value Bets Encontrados*\n\n"
        message += f"📊 Partidos analizados: {fixtures_analyzed}\n"
        message += f"✅ Value bets: {len(value_bets)}\n\n"

        for i, bet in enumerate(value_bets[:5], 1):
            message += f"*{i}. {bet.match}*\n"
            message += f"   🎲 {bet.market} → {bet.selection}\n"
            message += f"   💰 Cuota: {bet.odds_decimal:.2f}\n"
            message += f"   📈 EV: {bet.expected_value:+.2%}\n"
            message += f"   💵 Kelly: {bet.kelly_stake_pct:.2%}\n\n"

        if len(value_bets) > 5:
            message += f"_...y {len(value_bets) - 5} más_\n\n"

        message += "_Generado automáticamente por Sports Betting Bot_"

        return self.send_message(message)

    def send_settlement_notification(
        self,
        match: str,
        selection: str,
        result: str,
        pnl: float,
    ) -> bool:
        """
        Send notification for a settled bet.

        Args:
            match: Match description
            selection: Bet selection
            result: Bet result (WON/LOST/VOID)
            pnl: Profit/loss amount

        Returns:
            True if sent successfully
        """
        result_emoji = "✅" if result == "WON" else "❌" if result == "LOST" else "⚪"
        pnl_emoji = "💰" if pnl >= 0 else "💸"

        message = (
            f"{result_emoji} *Apuesta Liquidada*\n\n"
            f"📊 Partido: {match}\n"
            f"✅ Selección: {selection}\n"
            f"🏁 Resultado: {result}\n"
            f"{pnl_emoji} P&L: ${pnl:+.2f}"
        )

        return self.send_message(message)


def main() -> None:
    """Test notifier."""
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

    notifier.send_performance_report(test_analytics)


if __name__ == "__main__":
    main()
