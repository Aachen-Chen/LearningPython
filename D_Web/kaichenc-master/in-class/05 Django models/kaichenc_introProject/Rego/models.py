from django.db import models

# Create your models here.

class Student(models.Model):
    andrewID = models.CharField(max_length=10)
    fn = models.CharField(max_length=10)
    ln = models.CharField(max_length=10)

class Course(models.Model):
    courseID = models.CharField(max_length=6)
    cn = models.CharField(max_length=30)
    instructor = models.CharField(max_length=30)
    student = models.ManyToManyField(
        Student,
        # through='Enrollment',
        # through_fields=('student', 'course'),
    )

# class Enrollment(models.Model):
#     student = models.ForeignKey(Student, on_delete=models.CASCADE)
#     course = models.ForeignKey(Course, on_delete=models.CASCADE)

