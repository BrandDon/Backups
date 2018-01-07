import csv
import re
import traceback

import requests
from bs4 import BeautifulSoup, Tag


class User(object):
    def __init__(self):
        pass


class Card(object):
    def __init__(self):
        pass


class Deck(object):
    def __init__(self):
        self.cards = []
        self.hero = None
        self.upvotes = None
        self.type = None
        self.crafting_cost = None
        self.archetype = None
        self.mana_bars = {}
        self.user = User()
        self.url = None
        self.comments = None


class PageGetter(object):
    def __init__(self):
        self.page = None
        self.deck = Deck()

    def run(self, url):
        try:
            self.page = requests.get(url).content
            self.page = BeautifulSoup(self.page, 'html.parser')
        except Exception:
            traceback.print_exc()

        self.get_mana_scale(self.page)
        print self.deck.mana_bars

    def get_upvotes(self):
        self.deck.upvotes = self.page.find('div',
                                           {
                                               'class': 'rating-sum rating-average rating-average-ratingPositive tip'}).text
        if self.deck.upvotes[0] == u'-':
            self.deck.upvotes = -1 * int(self.deck.upvotes[1:])
        elif self.deck.upvotes[0] == u'+':
            self.deck.upvotes = int(self.deck.upvotes[1:])
        else:
            self.deck.upvotes = int(self.deck.upvotes)

    def get_mana_scale(self, page):
        mana = []
        mana_bars = page.find_all('li', attrs={'id': re.compile('deck-bar-\d')})
        for bar in mana_bars:
            amount = int(bar['data-count'])
            mana.append(amount)
        return mana


class PageRunner(object):
    def __init__(self):
        self.url = 'http://www.hearthpwn.com/decks?filter-deck-tag=5&filter-deck-type-op=3&filter-deck-type-val=10&page='
        self.deck_types = ['Control', 'Aggro', 'Mid-range', 'Midrange']
        self.pageGetter = PageGetter()

    def get_table(self, page):
        table = page.find('tbody')
        with open('decks.csv', 'ab') as database:
            writer = csv.writer(database)
            for listing in table.contents:
                stats = self.filter_listing(listing)
                if stats[0] is False:
                    continue
                else:
                    writer.writerow(stats)
            database.flush()
            return True

    def run(self):
        for i in xrange(1, 11976):
            page = requests.get(self.url + str(i)).content
            page = BeautifulSoup(page, 'html.parser')
            table = self.get_table(page)

    def filter_listing(self, listing):
        if type(listing) != Tag:
            return False, "not a tag"
        if listing.name != 'tr':
            return False, "tag not a tr"
        for column in listing:
            if 'col-deck-type' not in column.attrs['class']:
                continue
            else:
                for deck_type in self.deck_types:
                    if deck_type in column.text:
                        stats = self.pageGetter.get_mana_scale(listing)
                        if sum(stats) != 30:
                            return False, "deck size is not 30"
                        stats.append(deck_type)
                        return stats
                return False, ""
        return False, "NO DECK TYPE"


PageRunner().run()
