import typing
from dataclasses import dataclass

COLORS = ('white', 'black', 'grey', 'blue', 'green', 'red', 'purple', 'pink', 'yellow', 'orange', 'aqua', 'brown')
WHITELIST = ('solo',)
BLACKLIST = ('from_behind', 'comic', 'animated')


def accept(substring: str, tags: typing.List[str]) -> bool:
    required_tags = substring.split()
    ok = False
    for required_tag in required_tags:
        if required_tag.startswith('-'):
            ok = required_tag not in tags
        else:
            ok = required_tag in tags
    return ok


@dataclass
class Category:
    name: str
    _subcategories: typing.Optional[typing.Tuple[typing.Union[str, "Category"], ...]] = None
    blacklist: typing.Tuple[str, ...] = ()
    priority: int = 0

    def __len__(self) -> int:
        return len(self._subcategories)

    @property
    def subcategories(self) -> typing.Tuple[typing.Union[str, "Category"], ...]:
        if self._subcategories is None:
            self._subcategories = (self.name,)
        return self._subcategories

    @property
    def labels(self)-> typing.List[str]:
        return [(subcategory.name if isinstance(subcategory, Category) else subcategory) for subcategory in
                self.subcategories]

    def get_tags(self) -> str:
        tags = ' '.join((f'-{blacklisted_tag}' for blacklisted_tag in self.blacklist))
        for subcategory in self.subcategories:
            if isinstance(subcategory, str):
                tags += " " + subcategory
            else:
                tags += " " + subcategory.get_tags()
        return tags

    def get_request_tags(self) -> typing.List[str]:
        tags = []
        for subcategory in self.subcategories:
            if isinstance(subcategory, str):
                tags.append(" -".join((subcategory, *self.blacklist)))
            else:
                for request_tag in subcategory.get_request_tags():
                    tags.append(" -".join((request_tag, *self.blacklist)))
        return tags

    def accept(self, tags: typing.List[str]) -> typing.Optional[typing.Tuple[int, str]]:
        accepted = {}
        if any(blacklisted_tag in tags for blacklisted_tag in self.blacklist):
            return None
        for subcategory in self.subcategories:
            if (isinstance(subcategory, Category) and subcategory.accept(tags)) \
                    or (isinstance(subcategory, str) and accept(subcategory, tags)):
                accepted[self.priority] = accepted.get(self.priority, []) + [subcategory]

        if accepted:
            accepted_category = accepted[max(accepted.keys())][-1]
            accepted_name = accepted_category.name if isinstance(accepted_category, Category) else accepted_category
            for i, subcategory in enumerate(self.subcategories):
                subname = subcategory.name if isinstance(subcategory, Category) else subcategory
                if subname == accepted_name:
                    return i, subname
        return None


TOP_TYPES = Category(
    'regular_top_types',
    ('bodysuit', 'jacket', 'sweater', 'shirt', 'tank_top', 'dress', 'bra'),
    blacklist=('bra',)
)
BOTTOM_TYPES = Category(
    'regular_bottom_types',
    ('bodysuit', 'shorts', 'pants', 'skirt', 'dress', 'bikini', 'panties', 'buruma')
)

categories: typing.List[Category] = [
    Category(
        'gender',
        (
            Category('1girl', blacklist=('1boy', 'multiple_boys')),
            Category('1boy', blacklist=('1girl', 'multiple_girls')),
        )
    ),
    Category(
        'rating',
        (
            Category('safe', ('rating:s', 'rating:g')),
            Category('questionable', ('rating:q',)),
            Category('explicit', ('rating:e',)),
        )),
    Category(
        'hair_length',
        (
            Category('short_hair'),
            Category('medium_hair'),
            Category('long_hair'),
            Category('very_long_hair', priority=1)
        )
    ),
    Category(
        'hair_style',
        ('twintails', 'high_ponytail', 'side_ponytail', 'braided_ponytail', 'double_bun', 'half_updo', 'straight_hair')
    ),
    Category(
        'hair_color',
        tuple(Category(f'{color}_hair') for color in COLORS),
        blacklist=('multicolored_hair', 'streaked_hair')
    ),
    Category(
        'eye_color',
        (*(Category(f'{color}_eyes') for color in COLORS), "closed_eyes"),
    ),
    Category('sleeves_color',
             tuple(Category(f'{color}_sleeves') for color in COLORS),
             ),
    Category(
        'sleeves_length',
        (
            "sleeveless",
            Category('short_sleeves', priority=1),
            Category('long_sleeves', priority=1),
        )),
    Category(
        'navel',
        (
            Category('navel'),
            Category('covered_navel'),
            Category('no_navel', ('-covered_navel -navel',))
        )),
    Category(
        'breast_peek',
        (
            Category('nipples'),
            Category('cleavage'),
            Category('sideboob'),
            Category('backboob'),
            Category('underboob'),
            Category('no_breasts', ('-nipples -cleavage -sideboob -backboob -underboob',))
        )),
    Category(
        'nipples',
        (
            Category('nipples'),
            Category('covered_nipples'),
            Category('no_nipples', ('-covered_nipples -nipples',)),
        )),
    Category(
        'top_types',
        (
            TOP_TYPES,
            Category('open_top', ('open_shirt', 'open_bodysuit'), priority=1),
            Category('minimal_top', ('bra', 'bikini'), priority=3),
            Category('strapless', priority=2),
            Category('topless', ('topless', 'nude'), )
        )
    ),
    Category(
        'top_color',
        tuple(
            Category(color, tuple(f'{color}_{top_type}' for top_type in TOP_TYPES.subcategories))
            for color in COLORS
        )
    ),
    Category(
        'bottom_type',
        (
            Category('long_bottom', ('pants', 'long_dress', 'long_skirt', 'overalls')),
            Category('medium_bottom', ('shorts', 'medium_skirt', 'medium_dress', 'overall_shorts')),
            Category('short_bottom', ('short_shorts', 'short_dress', 'miniskirt'), priority=1),
            Category('very_short_bottom', ('microshorts', 'microskirt', 'buruma', 'bikini', 'panties'), priority=2),
            Category('bottomless', ('bottomless', 'nude')),
        ),
        blacklist=('shorts_under_skirt', 'pants_under_skirt', 'shorts_under_shorts')
    ),
    Category(
        'bottom_color',
        tuple(
            Category(color, tuple(f'{color}_{bottom_type}' for bottom_type in BOTTOM_TYPES.subcategories))
            for color in COLORS
        )
    ),
    Category('skirt_type', ('pleated_skirt', 'pencil_skirt')),
    Category('neckwear', ('necktie', 'neckerchief', 'bowtie')),
    Category('expressions', ('angry', 'annoyed', 'sad', 'scared', 'surprised', 'grin')),
    Category(
        'composition',
        ('portrait', 'upper_body', 'cowboy_shot', 'feet_out_of_frame', 'full_body')
    ),
    Category(
        'position',
        ('standing', 'sitting', 'squatting', 'kneeling', 'on_back', 'on_side', 'on_stomach')
    )
]
