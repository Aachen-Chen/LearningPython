from django.apps import AppConfig


class CalculatorConfig(AppConfig):
    name = 'calculator'

# def calculator():
#     cur = 0.0
#     pre = 0.0
#     opt = 0
#     while(True):
#         print("--- Current Screen -----")
#         print("    %.5f" %cur)
#
#         num = ord(input("One click: "))
#         if(cur==0):
#             while(num==45):
#                 print("--- Current Screen -----")
#                 print("        %.5f" % cur)
#                 num = ord(input("One click: "))
#                 if( (num < 58 and num > 47) or num==46):
#                     cur, opt = neg_opt(cur, opt, num)
#                     print("***", cur, opt, num)
#                     break
#             if(num < 58 and num > 48):
#                 print("***", cur, opt, num)
#                 cur, opt = neg_opt(cur, opt, num)
#         elif(cur<0):
#             if ( (num < 58 and num > 47) or num==46):
#                 cur, opt = neg_opt(cur, opt, num)
#         elif(cur>0):
#             cur, opt = pos_opt(cur, opt, num)
#         print(cur, opt, num)
#         continue
#
#
# def pos_opt(cur, opt, num):
#     if opt < 0:
#         if num < 58 and num > 48:
#             cur += (num - 48) * (10 ** opt)
#             opt -= 1
#         elif num == 46:
#             return cur, opt
#         elif num == 48:
#             opt -= 1
#         else:
#             print("Invalid input.")
#             return cur, opt
#     if opt == 0:
#         if num == 46:
#             opt = -1
#             return cur, opt
#         if cur == 0:
#             if num < 58 and num > 48:
#                 cur = num - 48
#             else:
#                 if num != 48:
#                     print("Invalid input.")
#                     return cur, opt
#         else:
#             if num < 58 and num > 47:
#                 cur = cur * 10 + num - 48
#             else:
#                 print("Invalid input.")
#                 return
#     return cur, opt
#
#
# def neg_opt(cur, opt, num):
#     if opt < 0:
#         if num < 58 and num > 48:
#             print("***", cur, opt, num)
#             cur -= (num - 48) * (10 ** opt)
#             opt -= 1
#         elif num == 46:
#             return cur, opt
#         elif num == 48:
#             opt -= 1
#         else:
#             print("Invalid input.")
#             return cur, opt
#     if opt == 0:
#         if num == 46:
#             opt = -1
#             return cur, opt
#         if cur == 0:
#             if num < 58 and num > 48:
#                 cur = -(num - 48)
#             else:
#                 if num != 48:
#                     print("Invalid input.")
#                     return cur, opt
#         else:
#             if num < 58 and num > 47:
#                 cur = cur * 10 - (num - 48)
#             else:
#                 print("Invalid input.")
#                 return
#     return cur, opt
#
# if __name__ == "__main__":
#     calculator()
