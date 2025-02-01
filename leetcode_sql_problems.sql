
'''1693. Daily Leads and Partners '''
https://leetcode.com/problems/daily-leads-and-partners/

# Write your MySQL query statement below
select date_id, 
    make_name , 
    count(distinct lead_id) as unique_leads, 
    count(distinct partner_id) as unique_partners 
from dailysales
group by date_id, make_name;


'''1683. Invalid Tweets'''
https://leetcode.com/problems/invalid-tweets/

# Write your MySQL query statement below
select tweet_id 
from tweets 
where length(content) > 15 ; 



''' 1667. Fix Names in a Table'''
https://leetcode.com/problems/fix-names-in-a-table/

# Write your MySQL query statement below
select user_id, 
    concat(upper(substring(name, 1,1)) , lower(substring(name,2))) as name 

from users 
order by user_id ; 
