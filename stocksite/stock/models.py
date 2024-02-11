from django.db import models


class Stock(models.Model):
    date = models.DateField('Date')
    trade_code = models.CharField('Trade Code', max_length=100)
    high = models.FloatField('High')
    low = models.FloatField('Low')
    open = models.FloatField('Open')
    close = models.FloatField('Close')
    volume = models.IntegerField('Volume')

    def __str__(self):
        return self.trade_code

