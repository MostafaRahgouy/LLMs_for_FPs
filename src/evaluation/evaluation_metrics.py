import re
import numpy as np
import operator
import math


def get_program_performance(inputs, label_outputs, pred_outputs, pint, is_task_3=False):
    assert len(inputs) == len(label_outputs) or len(inputs) == len(pred_outputs) or len(label_outputs) == len(
        pred_outputs)

    program_score, valid_score, facts_score = 0, 0, 0
    for inp, l_out, p_out in zip(inputs, label_outputs, pred_outputs):

        try:

            if not is_task_3:
                facts = inp.split('==;;')[:1][0].replace('FACT', 'CONTEXTS').replace(' =; ', '=')
                pred_p = p_out.replace(' =; ', '=')
                label_p = l_out.replace(' =; ', '=')

            else:
                p_out = p_out.replace('PROGRAM:', "")
                facts, pred_p = 'CONTEXTS: ', 'PROGRAM: '
                for item in p_out.split(' =; '):
                    if 'F' == item.strip()[0]:
                        facts += item + '='
                    else:
                        pred_p += item + '='
                pred_p = pred_p.rstrip("=")
                facts = facts.rstrip('=')

                label_p = []
                l_out = l_out.replace('PROGRAM:', "")
                for idx, item in enumerate(l_out.split(' =; ')):
                    if item.strip()[0] != 'F':
                        label_p.append(item)
                label_p = '='.join(label_p)
                label_p = 'PROGRAM: ' + label_p


            pred_program_out = compile_fp(facts, pred_p, pint)
            facts_score += pred_program_out['num_fact_score']

            label_program_out = compile_fp(facts, label_p, pint)

            if pred_program_out and pred_program_out['P']:
                valid_score += 1
                program_score += get_direct_score(label_program_out['P'], pred_program_out['P'], pint)

        except:
            continue

    program_score /= len(inputs)
    facts_score /= valid_score
    valid_score /= len(inputs)

    return round(program_score, 2), round(valid_score, 2), round(facts_score, 2)


def get_program_score(input, label_program, pred_program, pint, is_task_3=False):
    program_score, valid, facts_score = 0, 0, 0
    try:

        if not is_task_3:
            facts = input.split('==;;')[:1][0].replace('FACT', 'CONTEXTS').replace(' =; ', '=')
            pred_p = pred_program.replace(' =; ', '=')
            label_p = label_program.replace(' =; ', '=')

        else:
            p_out = pred_program.replace('PROGRAM:', "")
            facts, pred_p = 'CONTEXTS: ', 'PROGRAM: '
            for item in p_out.split(' =; '):
                if 'F' == item.strip()[0]:
                    facts += item + '='
                else:
                    pred_p += item + '='
            pred_p = pred_p.rstrip("=")
            facts = facts.rstrip('=')

            label_p = []
            l_out = label_program.replace('PROGRAM:', "")
            for idx, item in enumerate(l_out.split(' =; ')):
                if item.strip()[0] != 'F':
                    label_p.append(item)
            label_p = '='.join(label_p)
            label_p = 'PROGRAM: ' + label_p

        pred_program_out = compile_fp(facts, pred_p, pint)
        facts_score += pred_program_out['num_fact_score']

        label_program_out = compile_fp(facts, label_p, pint)

        if pred_program_out and pred_program_out['P']:
            valid = 1
            program_score = get_direct_score(label_program_out['P'], pred_program_out['P'], pint)
    except:
        return program_score, valid, facts_score
    return program_score, valid, facts_score


def get_direct_score(label, pred, pint):
    la = convert_to_float(label, pint)
    pr = convert_to_float(pred, pint)
    return accuracy_metric(la, pr)


def accuracy_metric(y, y_hat):
    if y is None or y_hat is None:
        return 0
    if y < 0 or y_hat < 0:
        return 0
    if y == 0 and y_hat == 0:
        return 1
    elif y == 0 or y_hat == 0:
        return max(0, 1 - np.abs(np.log10(np.abs(y - y_hat))))
    try:
        return max(0, 3 - np.abs(np.log10(y / y_hat))) / 3
    except:
        return 0


def convert_to_float(item, pint):
    if type(item) in [int, float, np.float64, np.float32]:
        return item
    if isinstance(item, pint.Quantity):
        return item.m
    try:
        number = pint(item)
        if type(number) in [int, float, np.float64, np.float32]:
            return number
        else:
            return number.m
    except:
        return None


def compile_fp(context, p, pint):
    var = {
        'num_fact_score': []
    }
    answer_to_fact = {}
    match_number = re.compile('-?\ *[0-9]+\.*[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)*')
    context = context.replace('CONTEXTS:', '').strip().split('=')
    for fact in context:
        if fact == '':
            continue
        try:
            var[fact[:fact.index(':')]] = float(re.findall(match_number, fact[fact.index(':') + 1:])[0])
        except:
            try:
                var[fact[:fact.index(':')]] = None
            except:
                pass

    p = p.replace('PROGRAM:', '').strip().split('=')
    funcs = {'Mul': operator.mul, 'Div': operator.truediv, 'Add': operator.add, 'Sub': operator.sub,
             'Pow': operator.pow, 'Min': lambda *a: min(*a), 'Log': lambda *a: math.log(*a),
             'Fac': lambda a: math.factorial(a)}
    paren_match = re.compile('\(([^()]+)\)')
    for line in p:
        if line[0] == 'Q' or line[0] == 'P':
            try:
                lhs, rhs = line.split('->')
                lhs = lhs.strip()
                rhs = rhs.strip()
                new = False
            except:
                try:
                    lhs, rhs = line.split('—>')
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    new = False
                except:
                    lhs, rhs = line.split(':')
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    new = True
            if (lhs not in var and not new) or (lhs in var and new):
                return None
            if '-> A' in line and line[0] == 'Q' and new:
                return None
            if '-> A' in line and line[0] == 'P' and new:
                return None
            if '|' in rhs:
                answer, fact = rhs.split('|')
                answer = answer.strip()
                fact = fact.strip()
                try:
                    var['num_fact_score'].append(get_direct_score(var[answer], var[fact], pint))
                except:
                    var['num_fact_score'].append(0)
                answer_to_fact[answer] = fact[1]
                var[lhs] = var[answer]
            elif any(re.search(r'\b' + func + r'\b', line) for func in funcs):
                for func in funcs:
                    if func in line:
                        break
                parens = [i.strip() for i in re.findall(paren_match, line)[0].split(',')]
                in_parens = []
                for i in parens:
                    if i in var and 'Q' not in i:
                        return None
                    if i in var:
                        in_parens.append(var[i])
                    else:
                        in_parens.append(float(i))
                if type(in_parens[0]) == str or type(in_parens[1]) == str:
                    return None
                var[lhs] = funcs[func](*in_parens)
            else:
                var[lhs] = rhs
        elif line[0] == 'A':
            ureg_conv = pint(line[line.index(':') + 1:])
            var[line[:line.index(':')]] = ureg_conv
        else:
            return None
    var['answer_to_fact'] = answer_to_fact
    var['num_fact_score'] = np.mean(var['num_fact_score']) if var['num_fact_score'] else 0
    return var
