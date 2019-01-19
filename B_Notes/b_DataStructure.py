

def tuple_example():

    # TODO: what's the standard of "<" on tuple?
    t1 = (1, 'Erebor', 800.45)
    t2 = (2, 'Rivendell', 500.67)
    t3 = (1,)
    t4 = (2,)
    t5 = ("a", )

    print(t3 < t4)
    print(t1 < t4)
    print(t1 < t2)

    # print(t3 < t5)
    # TypeError: ''<' not supported between instances of 'int' and 'str'


    return

def list_example():

    # TODO: generate, concatenate, append
    # TODO: what data type does list support?
    homo_list = [12, 45, 900, 78, 34, 66, 17, 85]
    hetero_list = [10, 'foo', 1.3]
    tuple_list = [
        (1, 'Erebor', 800.45),
        (2, 'Rivendell', 500.67),
        (3, 'Shire', 900.12),
        (4, 'Mordor', 1112.30)
    ]

   # TODO: concatenate two lists
    print(homo_list + hetero_list)

    # TODO: what is * for list?
    print(homo_list * 2)

    # TODO: what operation is unsupported for list?
    # print([0] - [0])
    # print([0] / [0])

    # TODO: how to make [10, 11, 12], [0, 0, 0], random list between [0,1)?
    l1 = list(range(10, 12))
    l2 = [0] * 3
    # random: use numpy


    # TODO: slicing and dicing
    # TODO: get first 4 element
    l5 = list(range(10))
    print(l5[:5])

    # TODO: get last [7, 8]
    print(l5[-3:-1])

    # TODO: take every two elements
    print(l5[::2])

    # TODO: reverse a list
    print(l5[-1:0:-1])
    # what about this?
    print(l5[-1:0])


    # TODO: operation on list
    # TODO: can we sum hetero_list?
    # sum(hetero_list)
    # TypeError: unsupported operand type(s) for +: 'int' and 'str'

    # TODO: how to compare list?
    print([0, 0] == [0, 0])
    print([0, 0] < [0, 0, 0])
    print([0, 0] < [1, 0])
    print([0, 0, 0] < [1, 0])
    print([1, 0] > [0, 'a'])
    # print([0, 0] > [0, 'a'])
    # answer: compare each pair of elements; if win, stop;
    #   if one has no more elment, smaller.

    # TODO: Can we sort a hetero_list, find its min, max?
    # How is sorted() implemented?
    # sorted(hetero_list)
    # TypeError: ''<' not supported between instances of 'str' and 'int'
    print(sorted(tuple_list))
    print(min(homo_list))

    return

# tuple_example()
list_example()
