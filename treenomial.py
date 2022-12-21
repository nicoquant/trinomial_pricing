import numpy as np
from time import time
from dataclasses import dataclass
from datetime import date, timedelta
import time


class TypeOptionCallPut(Exception):
    pass


class TypeOption(Exception):
    pass


@dataclass
class Node(object):
    value_root: float
    child_up: float = None
    child_mid: float = None
    child_down: float = None
    up: float = None
    down: float = None
    proba_up: float = None
    proba_mid: float = None
    proba_down: float = None
    value_node: float = None


@dataclass
class NodeTruck(Node):
    parent: float = None


@dataclass
class Option:
    option_call_put: str
    option_type: str
    maturity_date: date
    strike: float

    def IsAmerican(self):
        return self.option_type == "American"

    def IsEuropean(self):
        return self.option_type == "European"

    def IsCall(self):
        return self.option_call_put == "Call"

    def IsPut(self):
        return self.option_call_put == "Put"

    def __post_init__(self):
        assert self.option_call_put in ["Call", "Put"], TypeOptionCallPut(
            "option_type_call_put variable has to be call or put"
        )
        assert self.option_type in ["European", "American"], TypeOptionCallPut(
            "option_type variable has to be European or American"
        )


@dataclass
class Market:
    rate: float
    volatility: float
    dividend: float
    dividend_date: date
    stock_price_initial: float

    def __post_init__(self):
        assert self.volatility > 0, ValueError("Volatility has to be higher than 0")
        assert self.stock_price_initial > 0, ValueError("Stock price has to be higher than 0")


@dataclass
class Tree(object):
    def __init__(self, date, step) -> object:
        if (globals().get("market") is None) and (globals().get("option") is None):
            print(
                "In case you are NOT using the treevsbs module. Please define market and option classes."
            )
            from class_for_time_computation import market, option
            global market, option

        self.N: int = (option.maturity_date - date).days
        self.date: date = date
        self.step: float = step
        self.point_in_time: float = 0
        self.dt: float = (self.N / 365) / step
        self.price_base: float = market.stock_price_initial
        self.root: float = self.price_base
        self.call_put: str = option.option_call_put
        self.alpha: float = np.exp(
            market.rate * self.dt + market.volatility * np.sqrt(3 * self.dt)
        )
        self.df = np.exp(-market.rate * self.dt)
        self.div_day = (market.dividend_date - date).days
        if market.dividend != 0:
            self.activate = True
        else:
            self.activate = False

    def __post_init__(self):
        assert self.dt > 0, ValueError("dt has to be higher than 0")
        assert self.N > 0, ValueError("N has to be higher than 0")

    def _design_node(self, curr):

        if curr.child_mid is None:
            if self.div_day < self.point_in_time and self.activate:
                curr.child_mid = NodeTruck(
                    curr.value_root * self.df ** (-1) - market.dividend
                )
            else:
                try:
                    if curr.down.child_up is None:
                        raise AttributeError("")
                    curr.child_mid = curr.down.child_up
                except AttributeError:
                    try:
                        if curr.up.child_down is None:
                            raise AttributeError("")
                        curr.child_mid = curr.up.child_down
                    except AttributeError:
                        curr.child_mid = NodeTruck(curr.value_root * self.df ** (-1))

        if curr.child_up is None:
            if curr.child_mid.up is None:
                curr.child_up = Node(curr.child_mid.value_root * self.alpha)
            else:
                curr.child_up = curr.child_mid.up
        if curr.child_down is None:
            if curr.child_mid.down is None:
                curr.child_down = Node(curr.child_mid.value_root / self.alpha)
            else:
                curr.child_down = curr.child_mid.down

        curr.child_mid.up = curr.child_up
        curr.child_mid.down = curr.child_down
        curr.child_up.down = curr.child_mid
        curr.child_down.up = curr.child_mid

        curr.child_mid.parent = curr
        return curr

    def _design_node_new(self, curr):

        if self.div_day < self.point_in_time and self.activate:
            fwd = curr.value_root * self.df ** (-1) - market.dividend
        else:

            curr = self._design_node(curr)
            return curr.child_mid

        try:
            pip = curr.down.child_mid
            if pip is None:
                raise AttributeError("")
        except AttributeError:
            pip = curr.up.child_mid

        while pip.value_root * (1 + self.alpha) / 2 < fwd:
            save = pip
            if pip.up is None:
                pip.up = Node(pip.value_root * self.alpha)
            pip.up.down = save
            pip = pip.up

        while round(fwd, 3) <= round(pip.value_root * (1 + 1 / self.alpha) / 2, 3):

            save = pip
            if pip.down is None:
                pip.down = Node(pip.value_root / self.alpha)
            pip.down.up = save
            pip = pip.down
            if round(pip.value_root, 3) == 0:
                break

        return pip

    def insert(self):
        if self.root == self.price_base:
            self.root = Node(self.price_base)
        if self.N > 0:
            self._design_node(self.root)
            self._attribute_proba(self.root)

            self.point_in_time += self.N / self.step
            self._expending_new(self.root)
        else:
            raise ValueError("N has to be higher than 0")

    def _expending_new(self, curr_node):
        while True:
            self.point_in_time += self.N / self.step
            curr_node = curr_node.child_mid
            self._design_node(curr_node)
            self._attribute_proba(curr_node)
            baseline = curr_node

            while curr_node.up is not None:
                curr_node = curr_node.up
                curr_node.child_mid = self._design_node_new(curr_node)
                curr_node.child_down = curr_node.child_mid.down
                curr_node.child_up = Node(curr_node.child_mid.value_root * self.alpha)
                curr_node = self._relate_nodes(curr_node)
                self._attribute_proba(curr_node)

            curr_node = baseline

            while curr_node.down is not None:
                curr_node = curr_node.down
                curr_node.child_mid = self._design_node_new(curr_node)
                curr_node.child_up = curr_node.child_mid.up
                curr_node.child_down = Node(curr_node.child_mid.value_root / self.alpha)
                curr_node = self._relate_nodes(curr_node)
                self._attribute_proba(curr_node)

            curr_node = baseline
            self._relate_nodes(curr_node)
            if self.div_day < self.point_in_time and self.activate:
                self.activate = False
                market.dividend = 0
            if self.point_in_time >= self.N:
                return

    @staticmethod
    def _relate_nodes(curr):
        curr.child_mid.up = curr.child_up
        curr.child_mid.down = curr.child_down
        curr.child_up.down = curr.child_mid
        curr.child_down.up = curr.child_mid
        return curr

    def _attribute_proba(self, curr):

        var = self._variance(curr.value_root)
        if self.div_day < self.point_in_time and self.activate:
            div = market.dividend
        else:
            div = 0

        mean = self._expected_values(curr, div)
        curr.proba_down = (
            curr.child_mid.value_root ** (-2) * (var + mean ** 2)
            - 1
            - (self.alpha + 1) * (mean / curr.child_mid.value_root - 1)
        ) / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        curr.proba_up = (
            curr.child_mid.value_root ** (-1) * mean
            - 1
            - (self.alpha ** (-1) - 1) * curr.proba_down
        ) / (self.alpha - 1)
        curr.proba_mid = 1 - curr.proba_down - curr.proba_up
        return curr

    def _variance(self, node_val):
        return (
            node_val ** 2
            * np.exp(2 * market.rate * self.dt)
            * (np.exp(market.volatility ** 2 * self.dt) - 1)
        )

    def _expected_values(self, nodes, div):
        return nodes.value_root * np.exp(market.rate * self.dt) - div

    @staticmethod
    def last_node(arbre):
        while True:
            if arbre.child_mid is None:
                return arbre
            else:
                arbre = arbre.child_mid

    def _compute_price(self, par):
        return (
            max(
                par.proba_down * par.child_down.value_node
                + par.proba_mid * par.child_mid.value_node
                + par.proba_up * par.child_up.value_node,
                0,
            )
            * self.df
        )

    @staticmethod
    def _payoff_last_row(last):
        baseline = last
        if option.IsCall():
            while last is not None:
                last.value_node = max(last.value_root - option.strike, 0)
                last = last.up
            last = baseline.down
            while last is not None:
                last.value_node = max(last.value_root - option.strike, 0)
                last = last.down
        elif option.IsPut():
            while last is not None:
                last.value_node = max(option.strike - last.value_root, 0)
                last = last.up
            last = baseline.down
            while last is not None:
                last.value_node = max(option.strike - last.value_root, 0)
                last = last.down
        return baseline

    def pricing_(self):
        last_prices = self.last_node(self.root)
        last_prices = self._payoff_last_row(last_prices)
        while True:
            if last_prices == self.root:
                self.df = np.exp(-market.rate * self.dt)
                return self._compute_price(last_prices)
            parent = last_prices.parent
            base = parent
            while parent is not None:
                if option.IsEuropean():
                    parent.value_node = self._compute_price(parent)
                elif option.IsAmerican():
                    if option.IsCall():
                        parent.value_node = max(
                            self._compute_price(parent),
                            parent.value_root - option.strike
                        )
                    else:
                        parent.value_node = max(
                            self._compute_price(parent),
                            option.strike - parent.value_root
                        )
                parent = parent.up

            parent = base
            while parent is not None:
                if option.IsEuropean():
                    parent.value_node = self._compute_price(parent)
                elif option.IsAmerican():
                    if option.IsCall():
                        parent.value_node = max(
                            self._compute_price(parent),
                            parent.value_root - option.strike,
                        )
                    else:
                        parent.value_node = max(
                            self._compute_price(parent),
                            option.strike - parent.value_root,
                        )
                parent = parent.down
            last_prices = base


if __name__ == "__main__":
    market = Market(
        rate=0.02,
        volatility=0.2,
        dividend=3,
        dividend_date=date(2023, 3, 1),
        stock_price_initial=100,
    )
    option = Option(
        option_call_put="Put",
        option_type="European",
        maturity_date=date(2023, 7, 1),
        strike = 101,
    )
    tree = Tree(date(2022, 9, 1), step=100)
    start = time.time()
    tree.insert()
    print(time.time() - start)
    pricing = tree.pricing_()
    print(time.time() - start)
