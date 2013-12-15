#def snapout(spans, start_index, end_index):
#    '''
#    <spans> is a list of spans. start_index and end_index are character indices.
#    This function finds the following list: [s : (s in spans) AND (start_index <= s[0]
#    <= end_index OR start_index <= s[1] <= end_index)]. It then returns two indices which
#    are references into <spans> that designate the first and last items in this list.
#    '''
#    # TODO: Think of a better name for this function and a better doc string
#    # NOTE NOTE NOTE: I think it should return the last index +1 maybe for consistency?
#    
#    first_index = _binarySearchForSpanIndex(spans, start_index, True)
#    second_index = _binarySearchForSpanIndex(spans, end_index, False)
#    return (first_index, second_index)

def slice(spans, start_char, end_char, return_indices = False):
    '''
    '''
    #TODO: write this doc string...
    first_index = _binarySearchForSpanIndex(spans, start_char, True)
    second_index = _binarySearchForSpanIndex(spans, end_char, False)
    if return_indices:
        return first_index, second_index + 1
    return spans[first_index:second_index+1]

def _binarySearchForSpanIndex(spans, index, first):
    '''
    Perform a binary search across the list of spans to find the index in the spans that
    corresponds to the given character index from the source document. The parameter <first>
    indicates whether or not we're searching for the first or second element of the spans.
    '''
    # clamps to the first or last character index
    if index >= spans[-1][1]:
        index = spans[- 1][1]
    elif index < 0:
        index = 0
    element = 0 if first else 1
    lower = 0
    upper = len(spans)-1
    prev_span_index = (upper+lower)/2
    cur_span_index = prev_span_index
    cur_span = spans[cur_span_index]
    while True:
        if cur_span[element] == index: # found the exact index
            return cur_span_index
        elif cur_span[element] > index: # we need to look to the left
            if lower >= upper:
                if element == 0:
                    return cur_span_index - 1
                else:
                    if index >= cur_span[0]:
                        return cur_span_index
                    elif index <= spans[cur_span_index-1][1]:
                        return cur_span_index - 1
                    else:
                        return cur_span_index
            prev_span_index = cur_span_index
            upper = cur_span_index - 1
            cur_span_index = (upper+lower)/2
            cur_span = spans[cur_span_index]
        elif cur_span[element] < index: # we need to look to the right
            if lower >= upper:
                if element == 0:
                    if index <= cur_span[1]:
                        return cur_span_index
                    else:
                        return cur_span_index + 1
                else:
                    return cur_span_index + 1
            prev_span_index = cur_span_index
            lower = cur_span_index + 1
            cur_span_index = (upper+lower)/2
            cur_span = spans[cur_span_index]

def _test():
    spans = [(0, 4), (5, 7), (8, 11), (12, 18), (19, 21), (22, 27), (28, 34), (34, 35), (36, 38), (39, 46), (47, 56), (57, 60), (61, 64), (65, 71), (72, 74), (75, 83), (84, 87), (88, 97), (98, 103), (104, 109), (110, 114), (115, 124), (125, 129), (130, 134), (135, 142), (142, 143), (144, 147), (148, 150), (151, 157), (158, 163), (164, 167), (168, 174), (175, 177), (178, 181), (182, 187), (187, 188), (189, 192), (193, 201), (202, 205), (206, 211), (212, 219), (220, 222), (223, 228), (229, 232), (233, 237), (238, 240), (241, 247), (248, 251), (252, 254), (255, 261), (261, 263), (264, 267), (268, 275), (276, 280), (280, 281), (282, 283), (284, 290), (291, 298), (299, 301), (302, 305), (306, 314), (315, 317), (318, 325), (326, 334), (335, 339), (340, 344), (345, 351), (352, 359), (360, 363), (364, 370), (371, 376), (377, 382), (383, 387), (388, 390), (391, 394), (395, 406), (408, 410), (411, 415), (416, 421), (422, 428), (429, 431), (432, 434), (435, 447), (447, 448), (449, 453), (454, 457), (458, 461), (462, 465), (466, 473), (474, 479), (479, 480), (481, 485), (486, 490), (491, 494), (495, 502), (503, 505), (506, 511), (512, 519), (520, 524), (525, 532), (533, 544), (545, 551), (551, 552), (553, 557), (558, 563), (564, 569), (570, 573), (574, 578), (578, 579), (580, 587), (588, 591), (592, 595), (596, 603), (604, 606), (607, 617), (617, 619), (619, 623), (624, 626), (627, 633), (634, 639), (640, 646)]
    print slice(spans, 9, 60, return_indices = True) #should return (2, 11)

if __name__ == "__main__":
    _test()
    
