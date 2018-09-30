# Create your models here.
from django.contrib.auth.models import User
from django.db import models


# CREATE in order. Same as in Oracle.

class Post(models.Model):
    body = models.CharField(max_length=42)
    author = models.ForeignKey(User, on_delete=models.CASCADE, default=2)
    pub_date = models.DateTimeField('date published')
    # image = models.FilePathField()
    def __str__(self):
        return self.body +" Posted at: "+ self.pub_date.strftime("%b %d %Y %H:%M:%S") + " by " + self.author.username

