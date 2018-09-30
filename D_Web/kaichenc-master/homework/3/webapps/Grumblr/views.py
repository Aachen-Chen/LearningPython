from django.shortcuts import render, get_object_or_404, get_list_or_404
from django.http import Http404
from django.utils import timezone

# Create your views here.
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate

from grumblr.models import Post


@login_required
def home(request):
    global_stream = Post.objects.order_by('-pub_date')[:]
    context = {"global_stream": global_stream}
    return render(request, 'grumblr/home.html', context)


@login_required
def post(request):
    if request.method=='GET':
        return redirect('grumblr:home')

    if not 'body' in request.POST or not request.POST['body']:
        msg = 'Invalid input occurs.'
        print("no body")

        global_stream = Post.objects.order_by('-pub_date')[:]
        context = {"global_stream": global_stream, "error": msg}
        return render(request, 'grumblr/home.html', context)

    elif len(request.POST['body']) > 42:
        msg = 'Excess max length. Please shorten your post.'
        print("too long")

        global_stream = Post.objects.order_by('-pub_date')[:]
        context = {"global_stream": global_stream, "error": msg}
        return render(request, 'grumblr/home.html', context)

    else:
        new_post = Post(body=request.POST['body'], author=request.user, pub_date=timezone.now())
        new_post.save()
        return redirect('grumblr:home')


@login_required
def profile(request, username):
    user = User.objects.get(username = username)
    personal_stream = Post.objects.filter(author = user).order_by('-pub_date')[:5]
    context = {
        "user": user,
        "personal_stream": personal_stream
    }
    return render(request, 'grumblr/profile.html', context)


def rego(request):
    context = {}
    msg = []

    if request.method=='GET':
        return render(request, 'grumblr/rego.html', context)

    # for k, v in request.POST.items():
    #     print(k, v)
    # print(request.POST.get('username') == None)

    if request.POST.get('username') == None or request.POST.get('username')=='':
        print("no username")
        msg.append('Please enter a username.')
    else:
        if len(User.objects.filter(username=request.POST.get('username'))) > 0:
            msg.append('Username taken. Please select another one.')
        else:
            context['username'] = request.POST['username']

    if request.POST.get('firstname') == None or request.POST.get('firstname')=='':
        msg.append('Please enter your first name.')

    if request.POST.get('lastname') == None or request.POST.get('lastname')=='':
        msg.append('Please enter your last name.')

    if request.POST.get('pwd1') == None or request.POST.get('pwd1') == '':
        msg.append('Please enter a password.')
    if request.POST.get('pwd2') == None or request.POST.get('pwd2') == '':
        msg.append('Please confirm your password.')

    if request.POST.get('pwd1') != None and request.POST.get('pwd2') != None \
        and request.POST.get('pwd1') != '' and request.POST.get('pwd2') != '' \
        and request.POST.get('pwd1') != request.POST.get('pwd2'):
        msg.append('Passwords do not match.')

    context['msg'] = msg

    if msg:
        return render(request, 'grumblr/rego.html', context)

    new_user = User.objects.create_user(
        username=request.POST['username'],
        password=request.POST['pwd1'],
        first_name = request.POST['firstname'],
        last_name = request.POST['lastname'],
    )
    new_user.save()

    new_user = authenticate(
        username = request.POST['username'],
        password = request.POST['pwd1']
    )

    login(request, new_user)

    return redirect('grumblr:home')


