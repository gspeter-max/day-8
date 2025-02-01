'''
Write a SQL query to find the top 5 countries with the highest average number of actions per user, considering only users who have performed at least 5 actions.
Evaluation Criteria
Correctness of the query
Efficiency of the query
Ability to handle large datasets 
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



