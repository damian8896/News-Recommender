from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from articleUpdater import news


def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(news.update, 'interval', hours=6, next_run_time=datetime.now())
    scheduler.start()