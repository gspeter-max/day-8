
'''
Write a SQL query to find the top 5 countries with the highest average number of actions per user,
 considering only users who have performed at least 5 actions.
'''


select u.country , 
	avg(action_count) as avg_action 

from users u 
join (
	select user_id , 
	count(user_id) as action_count 
	from users u 
	group by user_id 
	having count(user_id) >= 5 
	) a 
on u.user_id = a.user_id 
group by u.country 
order by avg_action desc 
limit 5 ; 


'''
Write a SQL query to find the top 3 ad campaigns with the highest return on ad spend (ROAS), \
considering only campaigns that have spent at least 50% of their budget. ROAS is calculated as the total revenue divided by the total cost.
Assume that the revenue is $10 per click.count'''


with temp_temp as (
  select  
    ai.campaign_id as campaign_id,
    (count(ac.click_id) * 10) as total_revenue,
    sum(ai.cost) as total_cost,
    (sum(ai.cost) / ap.budget) * 100 as total_spending
  from  
    ad_campaigns ap
  left join 
    ad_impressions ai ON ap.campaign_id = ai.campaign_id
  left join 
    ad_clicks ac ON ai.impression_id = ac.impression_id
  group by 
    ai.campaign_id
  having  
    total_spending >= 50
)
select 
  campaign_id,
  case 
    when total_cost = 0 then 0 
    else total_revenue / total_cost 
  end as  roas
from  
  temp_temp
order by 
  roas desc
limit 3;

'''
Write a SQL query to find the top 5 customers with the highest total spend, 
along with their total spend amount and the number of orders they have placed.count
'''

with temp_temp as (
	select customer_id , 
	sum(total_amount) as total_sending ,
	count(order_id ) as total_order_placed 
	from orders 
	group by customer_id
) 
select t.customer_id , c.name , t.total_spending , t.total_order_placed
from temp_temp t 
inner  join customer_id c on t.customer_id = o.customer_id
order by total_spending desc 
limit 5 ; 
