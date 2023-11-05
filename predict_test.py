#!/usr/bin/env python
# coding: utf-8

import json

import requests
    
url = 'http://localhost:9696/predict'

student_id = "stu-123"
student = {
"parental_level_of_education":"associates degree",
"sat_total_score":2015,
"parental_income":76369,
"college_gpa":3.4
}

# print(student)
response = requests.post(url, json=student).json()
print(response)

if response['graduate'] == True:
    print('sending tutoring session offer email to %s' % student_id)
else:
    print('not sending tutoring session offer email to %s' % student_id)
