select cast(date_trunc('month', date) as date) as time,
mode, 
(COUNT(*) filter (where status = 'Confirmed') * 100.0 / count(*))::float as percents
from new_payments
where mode not like 'Не определено'
group by mode, cast(date_trunc('month', date) as date)
order by time asc,  mode asc