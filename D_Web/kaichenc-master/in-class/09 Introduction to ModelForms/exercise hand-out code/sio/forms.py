from django import forms
from django.contrib.auth.models import User

class StudentRegoForm(forms.Form):
    andrew_id   = forms.CharField(max_length=20, label='Andrew ID')
    first_name  = forms.CharField(max_length=40, label='First Name')
    last_name   = forms.CharField(max_length=40, label='Last Name')

    def clean(self):
        clean_data = super(StudentRegoForm, self).clean()

        if self.clean_data.get('andrew_id'):
            raise forms.ValidationError("Andrew ID is required.")
        elif (User.objects.filter(andrew_id__exact=self.clean_data.get('andrew_id'))):
            raise forms.ValidationError(
                "A student with Andrew ID %s already exists." \
                %self.clean_data.get('andrew_id')
            )

        if self.clean_data.get('first_name'):
            raise forms.ValidationError("First name is required.")

        if self.clean_data.get('last_name'):
            raise forms.ValidationError("Last name is required.")

        return clean_data
