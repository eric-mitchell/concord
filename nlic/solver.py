from pysat.formula import IDPool, WCNFPlus
from pysat.examples.rc2 import RC2
from typing import List, Tuple, Callable
import numpy as np


def validate_probability(p):
    if not isinstance(p, float):
        p = float(p)
    assert p <= 1, f"p should be in [0, 1]; got {p}"
    assert p >= 0, f"p should be in [0, 1]; got {p}"
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p


def log_odds(p):
    # convert from a probability to a maxSAT weight
    # we use log odds ratio
    p = validate_probability(p)
    return np.log(p) - np.log(1 - p)


def log_inv_prob(p):
    p = validate_probability(p)
    return -np.log(1 - p)


class Solver:
    def __init__(self, beta=0.5, normalize=True,
                 statement_gamma: Callable = log_odds,
                 constraint_gamma: Callable = log_inv_prob):
        self.beta = beta  # confidence in QA model, must be between 0 and 1
        self.normalize = normalize  # default behavior is to normalize within a question
        self.statement_gamma = statement_gamma
        self.constraint_gamma = constraint_gamma

    def __call__(self, statement_groups: List[List[Tuple[str, float]]],
                 relations: List[Tuple[str, str, str, float]] = None,
                 verbose: bool = False, normalize=None, rm_self_entail=False,
                 entailment_correction=False, groups_are_unary: bool = False, true_statements=[],
                 top_k_relations: int = None, relation_types: List = None):
        '''
        `statement_groups` should be a list of lists of tuples of (statement, confidence):
        [
          [ ("The UK PM is Boris Johnson", 0.4), ("The UK PM is Theresa May", 0.5) ], # statement group 1 (i.e., question 1)
          [ ("Messi plays for PSG", 0.3), ("Messi plays for Barcelona", 0.6) ] # statement group 2 (i.e., question 2)
        ]
        `relations` should be from the output of NLIInferencer. Expects a list of tuples, where each tuple is:
            (statement_1, statement_2, RELATION_TYPE, confidence), where RELATION_TYPE may be 'entailment' or 'contradiction'
        `normalize` should be a bool indicating whether or not to normalize within an answer set for a question.
            "None" defaults to the value set during initialization, which is "True"
        `rm_self_entail` should be a bool indicating whether to allow both answers (currently limited to answer choice count of 2)
            to be correct within an answer set for a question during the solver if the two answers share an entailment. Default is "False"
        `entailment_correction` should be a bool indicating whether or not, for a given entailment [(-a, b), weight], the solver should add
            [(a), weight/2] and [(b), weight/2]. Default is "False"
        `groups_are_unary` -- omit 'at most one' constraints
        `top_k_relations` -- if not none, consider only the k most probable relations
        `relation_types` -- constraint types to consider. A value of None means do not filter relations by type.
        RETURN:
            a list of indices corresponding to the chosen statement/answer within each statement group; will have one int for each group in
            the `statement_groups` passed in
        '''

        if relations is None:
            relations = []

        if relation_types is not None:
            relations = [
                relation
                for relation in relations
                if relation[2] in relation_types
            ]

        if top_k_relations is not None:
            relations = sorted(relations, key=lambda r: r[-1], reverse=True)[:top_k_relations]

        if normalize is None:
            normalize = self.normalize

        # list(dict.fromkeys(items)) is a stable de-duplicator: https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
        unique_statement_groups = [list(dict.fromkeys(g)) for g in statement_groups]

        # default is to make sure probabilities sum to one within each group
        if normalize:
            unique_statement_groups = [[(s, c / sum([x[1] for x in g])) for (s, c) in g] for g in unique_statement_groups]

        # split list of 2-tuples (s, c) back into tuple of 2 lists
        unique_statements, unique_confidences = tuple(zip(*[(s, c) for g in unique_statement_groups for (s, c) in g]))

        pool = IDPool()
        cnf = WCNFPlus()
        for s in unique_statements:
            pool.id(s)

        negs = []

        def append_weighted_clause(clause, weight=None):
            if weight is None:
                cnf.append(clause)
            elif weight > 0:
                cnf.append(clause, weight=weight)
            elif weight < 0:
                negs.append((clause, -weight))

        for idx, s_i in enumerate(unique_statements):
            id_i = pool.id(s_i)
            conf_i = unique_confidences[idx]
            w_i = self.beta * self.statement_gamma(conf_i)
            append_weighted_clause([id_i], weight=w_i)

        """
            Adding true statements that we force to be true
        """
        for s_i in true_statements:
            cnf.append([pool.id(s_i)])

        entailment_list = []

        if self.beta < 1:
            for s_1, s_2, relation, prob in relations:
                id_1, id_2 = pool.id(s_1), pool.id(s_2)
                conf_re = (1 - self.beta) * self.constraint_gamma(prob)
                if relation == "entailment":
                    append_weighted_clause([-id_1, id_2], weight=conf_re)
                    if entailment_correction:
                        append_weighted_clause([id_1], weight=conf_re / 2)
                        append_weighted_clause([id_2], weight=conf_re / 2)
                    entailment_list.append((id_1, id_2))
                elif relation == "contradiction":
                    append_weighted_clause([-id_1, -id_2], weight=conf_re)
        else:
            if verbose:
                print(
                    "Skipping constraints because beta = 1; constraints only affect solution for beta < 1")

        # for every group of statements (all the different answers to each question), add one
        # constraint requiring AT LEAST ONE statement is true and one constraint requiring
        # that AT MOST one statement is true
        if not groups_are_unary:
            for g in statement_groups:
                grp_ids = list(set([pool.id(s) for (s, c) in g]))
                cnf.append(grp_ids)  # At *least* one should be true
                if rm_self_entail:
                    assert len(grp_ids) <= 2, "you cannot use rm_self_entail for answer space greater than length 2"
                    if len(grp_ids) == 2:  # condition to check if two identical statements for one question
                        if (grp_ids[0], grp_ids[1]) in entailment_list or (grp_ids[1], grp_ids[0]) in entailment_list:
                            print("Entailment skip active")
                            continue
                cnf.append((grp_ids, 1), is_atmost=True)  # At *most* one should be true

        cnf.nv = len(pool.id2obj)  # because pysat doesn't properly increment this in `normalize_negatives``
        cnf.normalize_negatives(negs)

        if verbose:
            print("Hard constraints", cnf.hard)
            print("Atm constraints", cnf.atms)  # Newly added
            print("Soft constraints",
                  list(zip(cnf.soft, [f"{w:0.3f}" for w in cnf.wght])))
            print(len(cnf.hard) + len(cnf.soft) + len(cnf.atms),
                  "total constraints")

        res = RC2(cnf, solver="minicard").compute()
        
        if res is None:
            print("*** IMPORTANT ***")
            print("THIS SHOULD BE VERY RARE PER BATCH.")
            hashing_counter = 1 # Need to be able to map back as well... Maybe not since it's just index based
            
            if relations is None:
                relations = []
            if top_k_relations is not None:
                relations = sorted(relations, key=lambda r: r[-1], reverse=True)[:top_k_relations]
            if normalize is None:
                normalize = self.normalize
            ## REBUILDING FOR EDGE CASE ##
            
            # list(dict.fromkeys(items)) is a stable de-duplicator: https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
            hashed_tracker = {} # Used to rebuild nli outputs as well
            sg_rebuilt = [] # The actual statement groups to be used later down the line
            group_tracker = {}
            
            for i in range(len(statement_groups)):
                goi = statement_groups[i]
                g_rebuilt = []
                for j in range(len(goi)):
                    soi = goi[j]
                    new_s = str(hashing_counter) + soi[0]
                    hashing_counter += 1
                    g_rebuilt.append((new_s, soi[1]))
                    if soi[0] not in hashed_tracker:
                        hashed_tracker[soi[0]] = []
                    hashed_tracker[soi[0]].append(new_s)
                for soi1 in g_rebuilt:
                    for soi2 in g_rebuilt:
                        if soi1[0] not in group_tracker:
                            group_tracker[soi1[0]]=[]
                        group_tracker[soi1[0]].append(soi2[0])
                sg_rebuilt.append(g_rebuilt)
            
            if len(sg_rebuilt) != len(statement_groups):
                raise NameError('IMPLEMENTATION ERROR 1')
                                     
            statement_groups = sg_rebuilt
                                     
            r_rebuilt = []
                                     
            for i in range(len(relations)):
                one_relation = relations[i]
                s1 = one_relation[0]
                s2 = one_relation[1]
                rt = one_relation[2]
                con = one_relation[3]
                for hashed_s1 in hashed_tracker[s1]:
                    no_comparison_group = group_tracker[hashed_s1]
                    for hashed_s2 in hashed_tracker[s2]:
                        if hashed_s2 not in no_comparison_group:
                            r_rebuilt.append((hashed_s1, hashed_s2, rt, con))
            
            r_rebuilt = list(set(r_rebuilt))
                                     
            relations = r_rebuilt
            
            ## REBUILDING FOR EDGE CASE ##
                                     
            unique_statement_groups = [list(dict.fromkeys(g)) for g in statement_groups]

            # default is to make sure probabilities sum to one within each group
            if normalize:
                unique_statement_groups = [[(s, c / sum([x[1] for x in g])) for (s, c) in g] for g in unique_statement_groups]

            # split list of 2-tuples (s, c) back into tuple of 2 lists
            unique_statements, unique_confidences = tuple(zip(*[(s, c) for g in unique_statement_groups for (s, c) in g]))

            pool = IDPool()
            cnf = WCNFPlus()
            for s in unique_statements:
                pool.id(s)

            negs = []

            def append_weighted_clause(clause, weight=None):
                if weight is None:
                    cnf.append(clause)
                elif weight > 0:
                    cnf.append(clause, weight=weight)
                elif weight < 0:
                    negs.append((clause, -weight))

            for idx, s_i in enumerate(unique_statements):
                id_i = pool.id(s_i)
                conf_i = unique_confidences[idx]
                w_i = self.beta * self.statement_gamma(conf_i)
                append_weighted_clause([id_i], weight=w_i)

            entailment_list = []

            if self.beta < 1:
                for s_1, s_2, relation, prob in relations:
                    id_1, id_2 = pool.id(s_1), pool.id(s_2)
                    conf_re = (1 - self.beta) * self.constraint_gamma(prob)
                    if relation == "entailment":
                        append_weighted_clause([-id_1, id_2], weight=conf_re)
                        if weird_entail:
                            append_weighted_clause([id_1], weight=conf_re / 2)
                            append_weighted_clause([id_2], weight=conf_re / 2)
                        entailment_list.append((id_1, id_2))
                    elif relation == "contradiction":
                        append_weighted_clause([-id_1, -id_2], weight=conf_re)
            else:
                if verbose:
                    print(
                        "Skipping constraints because beta = 1; constraints only affect solution for beta < 1")

            # for every group of statements (all the different answers to each question), add one
            # constraint requiring AT LEAST ONE statement is true and one constraint requiring
            # that AT MOST one statement is true
            if not groups_are_unary:
                for g in statement_groups:
                    grp_ids = list(set([pool.id(s) for (s, c) in g]))
                    cnf.append(grp_ids)  # At *least* one should be true
                    if rm_self_entail:
                        assert len(grp_ids) <= 2, "you cannot use rm_self_entail for answer space greater than length 2"
                        if len(grp_ids) == 2:  # condition to check if two identical statements for one question
                            if (grp_ids[0], grp_ids[1]) in entailment_list or (grp_ids[1], grp_ids[0]) in entailment_list:
                                print("Entailment skip active")
                                continue
                    cnf.append((grp_ids, 1), is_atmost=True)  # At *most* one should be true

            cnf.nv = len(pool.id2obj)  # because pysat doesn't properly increment this in `normalize_negatives``
            cnf.normalize_negatives(negs)

            if verbose:
                print("Hard constraints", cnf.hard)
                print("Atm constraints", cnf.atms)  # Newly added
                print("Soft constraints",
                      list(zip(cnf.soft, [f"{w:0.3f}" for w in cnf.wght])))
                print(len(cnf.hard) + len(cnf.soft) + len(cnf.atms),
                      "total constraints")

            res = RC2(cnf, solver="minicard").compute()

        if verbose:
            print()
            print("*************** Solution ***************")
            print(res)
            print("True  Node  Group  Prob  Statement")
            for x in res:
                statement = pool.obj(abs(x))
                prob_group = []
                for i in range(len(unique_statements)):
                    if unique_statements[i] == statement:
                        prob_group.append(round(unique_confidences[i], 3))
                unique_idx = unique_statements.index(statement)
                prob = unique_confidences[unique_idx]
                tf = "****" if x > 0 else "    "
                sg = [idx for idx in range(len(statement_groups)) if
                      statement in [x[0] for x in statement_groups[idx]]]
                # print(tf.ljust(5), str(abs(x)).ljust(5), str(sg).ljust(6), f"{prob:0.3f}", statement)
                print(tf.ljust(5), str(abs(x)).ljust(5), str(sg).ljust(6),
                      str(prob_group).ljust(7),
                      statement.ljust(9))  # Added newly

        to_return = []
        for group in statement_groups:
            any_were_true = False
            for idx, (statement, _) in enumerate(group):
                if pool.id(statement) in res:
                    if not groups_are_unary:
                        to_return.append(idx)
                    else:
                        to_return.append(True)
                    any_were_true = True
                    break
            if groups_are_unary and not any_were_true:
                to_return.append(False)

        return to_return


if __name__ == "__main__":
    statement_groups = [
        [("The UK PM is Boris Johnson", 0.45)],
        [("The UK PM is Theresa May", 0.65)],
        [("The UK PM is David Cameron", 0.75)],
        [("Boris Johnson lives at No. 10 Downing St.", 0.5)],
        # statement group 1 (i.e., question 1)
        # [("Messi plays for PSG", 0.3), ("Messi plays for Barcelona", 0.6)] # statement group 2 (i.e., question 2)
    ]
    constraints = [
        (
            "The UK PM is Boris Johnson",
            "The UK PM is Theresa May",
            "contradiction",
            0.8,
        ),
        (
            "The UK PM is David Cameron",
            "The UK PM is Theresa May",
            "contradiction",
            0.9,
        ),
        (
            "The UK PM is Boris Johnson",
            "Boris Johnson lives at No. 10 Downing St.",
            "entailment",
            0.9,
        ),
    ]
    solver = Solver()
    res = solver(
        statement_groups,
        relations=constraints,
        verbose=True,
        normalize=False,
        groups_are_unary=True,
        relation_types=["entailment"],
    )
    print(res)