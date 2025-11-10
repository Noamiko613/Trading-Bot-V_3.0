@echo off
REM Clear TRADE_LIVE environment variable and start bot in paper trading mode
echo ============================================================
echo FIXING PAPER TRADING MODE
echo ============================================================
echo.
echo Clearing TRADE_LIVE environment variable...
set TRADE_LIVE=
echo Done.
echo.
echo Forcing paper trading in .env file...
python force_paper_trading.py
echo.
echo ============================================================
echo Environment is now clean. You can start the bot:
echo   python auto_trader.py
echo ============================================================
pause
