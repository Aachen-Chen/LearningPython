from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.template import loader

from django import forms
from django.forms import widgets
from django.forms import fields

# Create your views here.
def hello(request):
    return HttpResponse("Hello World @ new route!!")


def notes(request):
    return render(request, 'Notes_17637.html')


# def subnotes(request):
#     return render(request, 'sub/Notes_17637.html')

user_list = []


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('possword')
        # print(username, password)
        # print(request.POST)
        for k, v in request.POST.items():
            print(k, v)
        temp = {'user': username, 'pwd': password}
        user_list.append(temp)

    list = [1, 2, 3, 4, 5]
    context = {
        'list': list,
    }
    # return render(request, 'login.html', context)
    return render(request, 'login.html', {'data': user_list})
    # return HttpResponse(render(request, 'login.html', {'data': user_list}))


def index(request):
    list = [1, 2, 3, 4, 5]
    # template = loader.get_template('calculator/index.html')
    context = {
        'list': list,
    }
    # return HttpResponse(template.render(context, request))
    return HttpResponse(request, 'calculator/index.html', context)

def testPost(request):
    # print("testPost")
    # if request.method=='POST':
    #     return cal(request)
    # else:
    return render(request, 'test/testPost.html')


"""
Case:
get
    no param    norm    "Check"
    params      error   "
post
    no param    error   "missing:...
    1-9+-*/     norm
    others      error   "invalid:...
"""

def cal(request):
    cur, mem, opt, new, display = str(0), str(0), str(0), str(0), str(0)
    param = {"cur": cur, "mem": mem, "opt": opt, "display": display}

    optset = set(['+', '-', '*', '/'])
    setvalid = set(optset)
    if request.method =='GET':
        if not request.GET:
            param['err'] = ""
            return render(request, 'calculator/cal.html', param)
        else:
            param['err'] = "Invalid GET input. Please just click."
            return render(request, 'calculator/cal.html', param)


    if request.method == 'POST':
        for k, v in request.POST.items():
            print(k, v)

        setvalid.add('0')
        opt = validate(request.POST.get('opt'), setvalid)

        setvalid = set(['+', '-', '*', '/', 'C', '.', '='])
        cur = validate(request.POST.get('cur'), setvalid, allowDigit=True)
        mem = validate(request.POST.get('mem'), setvalid, allowDigit=True)
        new = validate(request.POST.get('new'), setvalid, allowDigit=True)

        print(cur, opt, mem, new)
        print(cur and opt and mem and new)

    if not (cur and opt and mem and new):
        param['err'] = "Invalid PUT input. Please just click."
        return render(request, 'calculator/cal.html', param)

    param['err'] = ""

    if new == "C":
        param= {"cur": 0, "mem": 0, "opt": 0, "display": 0}
        param['err'] = "Cleared. Please continue."
    elif new in optset:
        if opt == "0":
            param = {"cur": 0, "mem": cur, "opt": new, "display": cur}
        elif opt == new:
            param = {"cur": 0, "mem": mem, "opt": new, "display": mem}
        else:
            result = comp(mem, opt, cur)
            if result=="Error":
                param = {"cur": 0, "mem": 0, "opt": 0, "display": "Error"}
            else:
                param = {"cur": 0, "mem": result, "opt": new, "display": round(result)}
    elif new == "=":
        if opt=="0":
            param = {"cur": cur, "mem": mem, "opt": 0, "display": cur}
        else:
            result = comp(mem, opt, cur)
            if result=="Error":
                param = {"cur": 0, "mem": 0, "opt": 0, "display": "Error"}
            else:
                param = {"cur": 0, "mem": 0, "opt": 0, "display": round(result)}
                param['err'] = "Calculated and cleared. Now start new calculation."
    else:
        digit = digitin(cur, new)
        param = {"cur": digit, "mem": mem, "opt": opt, "display": digit}


    return render(request, 'calculator/cal.html', param)


def digitin(old: str, new: str) -> str:
    if old == "0":
        old = new
    else:
        old = ''.join([old, new])
    return old


def comp(l: str, opt: str, r: str):
    if opt == "+":
        return float(l) + float(r)
    elif opt == "-":
        return float(l) - float(r)
    elif opt == "*":
        return float(l) * float(r)
    else:
        if r == "0":
            return 'Error'
        else:
            return float(l) / float(r)

def validate(obj, legalSet: set, allowDigit: bool = False):
    if obj==None:
        return False
    if allowDigit:
        if (obj in legalSet) \
            or (obj.count(".")<2 and obj.replace(".","").isdigit()):
            return obj
        else:
            return False
    else:
        if obj in legalSet:
            return obj
        else:
            return False
