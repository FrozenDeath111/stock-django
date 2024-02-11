from django.shortcuts import render, HttpResponse, redirect, HttpResponseRedirect
import datetime
import plotly.express as px
import plotly.graph_objects as pgo
import pandas as pd
from .utils import data_handle, Test_ML
from .models import Stock
from .forms import StockForm


# Create your views here.
def index_json(request):
    data = data_handle.handle_json()
    return render(request, 'index_json.html', {
        'data': data,
    })


def index_csv(request):
    data = data_handle.handle_csv()
    return render(request, 'index_csv.html', {
        'data': data,
    })


def index(request):
    data_from_db = Stock.objects.all().values()

    trade_code_set = set([item['trade_code'] for item in data_from_db])

    chart = ''
    trade_code = ''

    if request.method == "POST":
        chart_type = request.POST.get('chart_type')

        if chart_type == 'all':
            df = pd.DataFrame.from_records(data_from_db)
            y_axis = request.POST.get('y_axis')
            chart = charts('all', df, y_axis=y_axis)

        else:
            trade_code = request.POST.get('trade_code')
            if not trade_code:
                return HttpResponse('You need to select trade code for this chart')
            trade_data = Stock.objects.filter(trade_code=trade_code).values()
            df = pd.DataFrame.from_records(trade_data)

            y_axis = request.POST.get('y_axis')
            z_axis = request.POST.get('z_axis')

            charts_need_y_axis = ['line_chart', 'bar_chart', 'pie_chart']
            if chart_type in charts_need_y_axis:
                if not y_axis:
                    return HttpResponse('You need to select trade code and y-axis for this chart')

            charts_need_both_axis = ['bar_line_chart', '3d_chart',]
            if chart_type in charts_need_both_axis:
                if not y_axis and not z_axis:
                    return HttpResponse('You need to select trade code, y-axis and z-axis for this chart')

            chart = charts(chart_type, data=df, trade_code=trade_code, y_axis=y_axis, z_axis=z_axis)

    if request.method == "GET":
        df = pd.DataFrame.from_records(data_from_db)
        chart = charts('all', df, y_axis='close')

    return render(request, 'index.html', {
        'data': data_from_db,
        'trade_code': trade_code_set,
        'chart': chart,
        'selected_trade_code': trade_code
    })


def charts(chart_type, data, trade_code=None, y_axis=None, z_axis=None):
    if chart_type == 'candlestick':
        fig = pgo.Figure(
            data=pgo.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
        )

        fig.update_layout(
            title=trade_code,
            yaxis_title=trade_code + " Stock"
        )

    elif chart_type == 'line_chart':
        fig = px.line(
            data,
            x='date',
            y=y_axis,
            title=trade_code
        )

    elif chart_type == 'bar_chart':
        fig = px.bar(
            data,
            x='date',
            y=y_axis,
            title=trade_code
        )

    elif chart_type == 'bar_line_chart':
        fig = pgo.Figure()

        fig.add_trace(
            pgo.Scatter(
                x=data['date'],
                y=data[y_axis],
                name=y_axis
            )
        )

        fig.add_trace(
            pgo.Bar(
                x=data['date'],
                y=data[z_axis],
                name=z_axis
            )
        )

        fig.update_layout(
            title=trade_code
        )

    elif chart_type == '3d_chart':
        fig = pgo.Figure(
            data=[
                pgo.Mesh3d(
                    x=data['date'],
                    y=data[y_axis],
                    z=data[z_axis],
                    opacity=0.5,
                    color='rgba(255, 0, 0, 0.5)'
                )
            ]
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='Date',
                yaxis_title=y_axis,
                zaxis_title=z_axis,
            )
        )

    elif chart_type == 'pie_chart':
        fig = px.pie(
            data,
            values=y_axis,
            names='date',
            title=trade_code
        )

    elif chart_type == 'sun_burst_chart':
        fig = px.sunburst(
            data,
            path=['date', 'high', 'low', 'open', 'close'],
            values='volume',
            title=trade_code
        )
    else:
        if not y_axis:
            y_axis = 'close'

        fig = px.line(
            data,
            x='date',
            y=y_axis,
            color='trade_code',
            title='Showing All Stock Data'
        )

    return fig.to_html()


def add_data(request):
    submitted = False
    if request.method == "POST":
        form = StockForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/add-data?submitted=True')
    else:
        form = StockForm
        if 'submitted' in request.GET:
            submitted = True

    return render(request, 'add_data.html', {
        'form': form,
        'submitted': submitted
    })


def pred_data(request):
    data_from_db = Stock.objects.all().values()

    trade_code_set = set([item['trade_code'] for item in data_from_db])
    result = ''
    trade_code = ''
    pred_type = ''
    lookback_days = ''

    if request.method == "POST":
        trade_code = request.POST.get('trade_code')
        pred_type = request.POST.get('pred_type')
        lookback_days = request.POST.get('lookback_days')

        predicted_data = Test_ML.train_and_predict_next_day(trade_code, int(lookback_days), pred_type)

        result = predicted_data[0][0]

    return render(request, 'prediction_data.html', {
        'trade_code': trade_code_set,
        'selected_trade_code': trade_code,
        'pred_type': pred_type,
        'lookback_days': lookback_days,
        'result': result
    })


def edit_data(request, stock_id):
    stock = Stock.objects.get(pk=stock_id)
    form = StockForm(request.POST or None, instance=stock)
    if form.is_valid():
        form.save()
        return redirect('index')
    return render(request, 'edit_data.html', {
        'id': stock_id,
        'form': form
    })


def delete_data(request, stock_id):
    stock = Stock.objects.get(pk=stock_id)
    stock.delete()
    return redirect('index')


def save_data(request):
    data = data_handle.handle_json()
    try:
        for item in data:
            Stock.objects.create(
                date=datetime.datetime.strptime(item['date'], '%Y-%m-%d').date(),
                trade_code=item['trade_code'],
                high=float(item['high'].replace(',', '')),
                low=float(item['low'].replace(',', '')),
                open=float(item['open'].replace(',', '')),
                close=float(item['close'].replace(',', '')),
                volume=int(item['volume'].replace(',', ''))
            )
            # stock.date = datetime.datetime.strptime(item.date, '%m-%d-%Y').date()
            # stock.trade_code = item.trade_code
            # stock.high = float(item.high)
            # stock.low = float(item.low)
            # stock.open = float(item.open)
            # stock.close = float(item.close)
            # stock.volume = int(item.high.replace(',', ''))
    except Exception as ex:
        return HttpResponse(ex)

    return redirect('index')
