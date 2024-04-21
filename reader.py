import cv2
import pytesseract
from mrz.checker.td1 import TD1CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td3 import TD3CodeChecker
import pycountry
from datetime import datetime


class Reader:
    def __init__(self):
        # set white list chars to config
        white_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'

        self._config = f'tessedit_char_whitelist={white_list}'

        self._fields = [
            'surname',
            'name',
            'country',
            'nationality',
            'birth_date',
            'expiry_date',
            'sex',
            'document_type',
            'document_number',
            'optional_data',
            'birth_date_hash',
            'expiry_date_hash',
            'document_number_hash',
            'final_hash',
        ]

    def read_mrz(self, image):
        """
        Read mrz code from cropped image

        :param np.ndarray image: cropped image
        :return: extracted text
        :rtype: str
        """
        # extracted text from image
        extracted = pytesseract.image_to_string(
            image,
            lang='mrz',
            config=self._config
        )

        # split by enter char
        lines = extracted.split('\n')

        codes = []

        # remove text if line does not match pattern
        for line in lines:
            if len(line) >= 30:
                codes.append(line.replace(' ', ''))

        return '\n'.join(codes)

    def get_fields(self, code):
        """
        Extract identity fields from string

        :param str code: code contains identity fields
        :return: extracted fields
        :rtype: dict
        """
        result = {}

        try:
            # validate code
            # 92 characters -> ID card
            # 89 characters -> Passport
            # print("Length of Code: ", len(code))
            # print("Code: ", code)
            if len(code) == 92:
                checker = TD1CodeChecker(code)
            elif len(code) == 73:
                checker = TD2CodeChecker(code)
            elif len(code) == 89:
                checker = TD3CodeChecker(code)
            else:
                raise Exception('The MRZ code could not be detected.')
            print("Truth: ", bool(checker))
            # extract fields
            fields = checker.fields()

            # add field to result dict
            for field in self._fields:
                val = getattr(fields, field)
                result[field] = val


        except Exception as e:
            raise Exception(e) from e

        return result

    def formatCode(self, data):
        if data['document_type'][:1] == 'P':
            data['document_type'] = 'Passport'
        elif data['document_type'][:1] == 'I':
            data['document_type'] = 'ID Card'
        elif data['document_type'][:1] == 'V':
            data['document_type'] = 'Visa Card'

        country = pycountry.countries.get(alpha_3=data['country'])
        data['country'] = country.name
        country = pycountry.countries.get(alpha_3=data['nationality'])
        data['nationality'] = country.name

        if data['sex'] == 'M':
            data['sex'] = 'Male'
        elif data['sex'] == 'F':
            data['sex'] = 'Female'
        else:
            data['sex'] = 'Undefined'

        data['birth_date'] = datetime.strptime(data['birth_date'], '%y%m%d').strftime('%Y-%m-%d')
        data['expiry_date'] = datetime.strptime(data['expiry_date'], '%y%m%d').strftime('%Y-%m-%d')

        return data

    def process(self, image):
        """
        Extract identity fields from cropped image

        :param np.ndarray image: cropped image
        :return: identity fields as json format
        :rtype: json
        """
        # convert image to gray color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get text from cropped image
        code = self.read_mrz(gray)

        # get identity fields from text
        fields = self.get_fields(code)

        # return formatting code
        return self.formatCode(fields)
